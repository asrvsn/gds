''' Specifying partial differential equations on graph domains ''' 

import numpy as np
import networkx as nx
from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict
from scipy.integrate import DOP853, LSODA
from scipy.optimize import least_squares
from abc import ABC, abstractmethod
import pdb
from enum import Enum
import os.path
import os
import hickle as hkl
import cloudpickle
from tqdm import tqdm
import cvxpy as cp

from .types import *
from .utils import *

''' System objects ''' 

class System:
	def __init__(self, stepper: Steppable, observables: Dict[str, Observable]):
		self._stepper = stepper
		self._observables = observables

	@property
	def stepper(self):
		return self._stepper

	@property 
	def observables(self) -> Dict[str, Observable]:
		return self._observables

	def solve_to_disk(self, T: float, dt: float, folder: str, parent='runs'): 
		assert os.path.isdir(parent), f'Parent directory "{parent}" does not exist'
		path = parent + '/' + folder
		if not os.path.isdir(path):
			os.mkdir(path)
		dump = dict()
		obs_items = self.observables.items()
		for name, obs in obs_items:
			dump[name] = []
		t = 0.
		try:
			with tqdm(total=int(T / dt), desc=folder) as pbar:
				while t < T:
					self.stepper.step(dt)
					for name, obs in obs_items:
						dump[name].append(obs.y.copy())
					t += dt
					pbar.update(1)
		finally:
			# Dump simulation data
			for name, data in dump.items():
				hkl.dump(np.array(data), f'{path}/{name}.hkl', mode='w', compression='gzip')
			# Dump system object
			with open(f'{path}/system.pkl', 'wb') as f:
				self.dt = dt # Save the dt (hacky)
				cloudpickle.dump(self, f)

	@staticmethod
	def from_disk(folder: str, parent='runs'):
		path = parent + '/' + folder
		assert os.path.isdir(path), 'The given path does not exist'
		with open(f'{path}/system.pkl', 'rb') as f:
			sys = cloudpickle.load(f)
		data = dict()
		n = 0
		for name in sys.observables.keys():
			data[name] = hkl.load(f'{path}/{name}.hkl')
			n = data[name].shape[0]
		sys_dt = sys.dt

		class DummyIntegrable(Steppable):
			def __init__(self):
				self.t = 0.
				self.i = 0

			def step(self, dt: float):
				T = self.t + dt
				while self.t < T and self.i < n:
					self.t += sys_dt
					self.i += 1

			def reset(self):
				self.t = 0.
				self.i = 0

		integ = DummyIntegrable()
		for name, obs in sys.observables.items():
			obs.history = data[name] # Hacky
			attach_dyn_props(obs, {'y': lambda self: self.history[integ.i], 't': lambda _: integ.t})

		return System(integ, sys.observables)


''' Base class: dynamical system on arbitrary finite domain ''' 

class fds(Observable, Steppable):
	def __init__(self, 
			X: Domain, 
			nonnegative=False,
		):
		''' 
		''' 
		assert (dydt is not None or lhs is not None), 'Either pass a time-difference or LHS of equation to be satisfied'
		Observable.__init__(self, X)
		Steppable.__init__(self, IterationMode.none)
		self.dynamic_bc = True
		self.dirichlet = lambda t, x: None
		self.X_dirichlet = []
		self.dirichlet_indices = np.array([], dtype=np.intp)
		self.dirichlet_values = np.array([])
		self.neumann = lambda t, x: None
		self.X_neumann = []
		self.neumann_indices = np.array([], dtype=np.intp)
		self.neumann_values = np.array([])
		self.nonnegative = nonnegative

	''' Dynamics ''' 

	def set_evolution(self,
			dydt: Callable[[Time, np.ndarray], np.ndarray]=None, order: int=1, max_step: float=1e-3, solver_args: Dict={},
			cost: Callable[[Time, np.ndarray], np.ndarray]=None, 
			map_fun: Callable[[Time, np.ndarray], np.ndarray]=None, dt: float=1.0,
		): 
		''' Define evolution law for the dynamics.

		Option 1: As a PDE
			dydt: Callable[Time]
				RHS of differential equation [uses LSODA for stiffness detection; falls back to DOP853 if not available]
			order: int
				[optional] Order of time-difference; if greater than one, automatically creates (order*ndim) state vector
			max_step: float
				[optional] Maximum allowed step size in the solver; default 1e-3
			solver_args: Dict
				[optional] Additional arguments to be passed to the solver

		Option 2: As a convex program
			cost: Callable[time]
				RHS of a disciplined-convex cost function (either array-like or scalar)
			solver_args: Dict
				[optional] Additional arguments to be passed to the solver

		Option 3: As a recurrence relation
			map_fun: Callable[time]
				RHS of recurrence relation
			dt: float
				[default 1.0] time delta for stepping 
		'''
		assert oneof([dydt != None, cost != None, fun != None]), 'Exactly one evolution law must be specified'
		self.t0 = 0.
		if dydt != None:
			self.iter_mode = IterationMode.dydt
			self.dydt_fun = dydt
			self.max_step = max_step
			self.order = order
			self.solver_args = self.solver_args
			self.y0 = np.zeros(self.ndim*order)
			try:
				self.integrator = LSODA(self.dydt, self.t0, self.y0, np.inf, max_step=max_step, **solver_args)
			except:
				print('Failed to use LSODA, falling back to DOP853')
				self.integrator = RK45(lambda t, y: y0, self.t0, self.y0, np.inf, max_step=max_step, **solver_args)
				self.integrator.fun = self.dydt

		elif cost != None:
			self.iter_mode = IterationMode.cvx
			self.cost_fun = cost
			self.solver_args = self.solver_args
			self.y0 = np.zeros(self.ndim)
			self._t = self.t0
			self._y = self.y0.copy()
			self._t_prb = cp.Parameter(nonneg=True)
			self._y_prb = cp.Variable(self._y.size)
			_cost = cost(self._t_prb, self._y_prb)
			if _cost.shape != (): # Cost is not scalar
				_cost = cp.sum(cp.abs(_cost))
			assert _cost.is_dcp(), 'Problem is not disciplined-convex'
			self._prb = cp.Problem(cp.Minimize(_cost), [])

		elif map_fun != None:
			self.iter_mode = IterationMode.map
			self.map_fun = map_fun
			self.y0 = np.zeros(self.ndim)
			self._t = self.t0
			self._n = int(self._t)
			self._y = self.t0.copy()

	def set_initial(self, 
			t0: float=0., 
			y0: Union[Callable[[Point], float], np.ndarray]=lambda _: 0.,
		):
		''' Set initial conditions. 
			t0: float
				Starting time
			y0: Union[Callable[[Point], float], np.ndarray]
				Function or array of points specifying intial condition.

		TODO: support for higher-order initial conditions in differential equations.
		'''
		assert self.iter_mode != IterationMode.none, 'Use set_evolution() before setting initial conditions'
		if type(y0) is np.ndarray:
			y0 = lambda x: y0[self.X[x]]
		self.t0 = t0
		self.y0_fun = y0

		if self.iter_mode is IterationMode.dydt:
			self.integrator.t = t0
		elif self.iter_mode is IterationMode.cvx:
			self._t = t0
		elif self.iter_mode is IterationMode.map:
			self._t = t0
			self._n = int(t0)

		for x in self.X.keys() - self.X_dirichlet:
			i, y = self.X[x], y0(x)
			self.y0[i] = y

			if self.iter_mode is IterationMode.dydt:
				self.integrator.y[i] = y
			elif self.iter_mode is IterationMode.cvx or self.iter_mode is IterationMode.map:
				self._y[i] = y

	def set_constraints(self, 
			dirichlet: Union[Callable[[Time, Point], float], Callable[[Point], float], Dict[Point, float]]={}, 
			neumann: Union[Callable[[Time, Point], float], Callable[[Point], float], Dict[Point, float]]={},
			project: Callable[[np.ndarray], np.ndarray]=lambda x: x,
		):
		''' Impose constraints. Assumes domain boundaries do not change.

		dirichlet: callable or dictionary
			Impose fixed values at boundary; either a function (t, x) or function (x) or static lookup dictionary
		neumann: callable or dictionary:
			Impose fixed fluxes at boundary; either a function (t, x) or function (x) or static lookup dictionary
		project: callable
			Project solutions onto feasible set.
		''' 
		assert self.iter_mode != IterationMode.none, 'Use set_evolution() before setting boundary conditions'
		if type(dirichlet) is dict:
			dirichlet = dict_fun(dirichlet)
		if type(neumann) is dict:
			neumann = dict_fun(neumann)
		self.dirichlet_fun = dirichlet
		self.neumann_fun = neumann
		self.project_fun = project

		# Store whether we will require recalculation
		self.dynamic_dirichlet = fun_ary(dirichlet) > 1 
		self.dynamic_neumann = fun_ary(neumann) > 1

		def populate(fun: Callable, dynamic: bool) -> Tuple[List, np.ndarray, np.ndarray]:
			if dynamic:
				domain = [x for x in self.X if (fun(0., x) is not None)]
				indices = np.array([self.X[x] for x in domain], dtype=np.intp)
				values = np.array([fun(0., x) for x in domain])
				return domain, indices, values
			else:
				domain = [x for x in self.X if (fun(x) is not None)]
				indices = np.array([self.X[x] for x in domain], dtype=np.intp)
				values = np.array([fun(x) for x in domain])
				return domain, indices, values

		# Store domain, indices, and values
		self.X_dirichlet, self.dirichlet_indices, self.dirichlet_values = populate(dirichlet, self.dynamic_dirichlet)
		self.X_neumann, self.neumann_indices, self.neumann_values = populate(neumann, self.dynamic_neumann)

		# Ensure nonoverlapping conditions
		intersect = set(self.X_dirichlet) & set(self.X_neumann)
		assert len(intersect) == 0, f'Dirichlet and Neumann conditions overlap on {intersect}'

		self.y0 = project(replace(self.y0, self.dirichlet_indices, self.dirichlet_values))

		if self.iter_mode is IterationMode.dydt:
			self.integrator.y[self.dirichlet_indices - self.ndim] = self.dirichlet_values
		elif self.iter_mode is IterationMode.cvx:
			self._y = self.y0.copy()
			self._y_cstr = cp.Parameter(self.dirichlet_values.size)
			self._y_cstr.value = self.dirichlet_values
			constr = [self._y_prb[self.dirichlet_indices] == self.y_cstr]
			self._prb = cp.Problem(self._prb.objective, constr)
		elif self.iter_mode is IterationMode.map:
			self._y = self.y0.copy()

	''' Stepping ''' 

	def reset(self):
		''' Reset the system to initial conditions ''' 
		assert self.iter_mode != IterationMode.none
		if self.iter_mode is IterationMode.dydt:
			self.set_evolution(dydt=self.dydt_fun, order=self.order, max_step=self.max_step, solver_args=self.solver_args)
		elif self.iter_mode is IterationMode.cvx:
			self.set_evolution(cost=self.cost_fun, solver_args=self.solver_args)
		elif self.iter_mode is IterationMode.map:
			self.set_evolution(map_fun=self.map_fun)
		self.set_initial(t0=self.t0, y0=self.y0_fun)

	def step(self, dt: float):
		''' Step the system to t+dt ''' 
		if self.iter_mode is IterationMode.none:
			raise Exception('Evolution law not specified')
		elif self.iter_mode is IterationMode.dydt:
			self.step_dydt(dt)
		elif self.iter_mode is IterationMode.cvx:
			self.step_cvx(dt)
		elif self.iter_mode is IterationMode.map:
			self.step_map(dt)
		else:
			raise Exception(f'Unsupported evolution law: {self.iter_mode}')

	''' Differential stepping ''' 

	def step_dydt(self, dt: float):
		self.integrator.t_bound = self.t + dt
		self.integrator.status = 'running'
		while self.integrator.status != 'finished':
			self.integrator.step()
			self.update_constraints(self.t)
			self.set_constraints()

	def dydt(self, t: Time, y: np.ndarray):
		self.update_constraints(t)
		n, order = self.ndim, self.order
		ret = np.zeros_like(y)
		for i in range(order-1):
			ret[n*i:n*(i+1)] = y[n*(i+1):n*(i+2)]
		diff = self.dydt_fun(t, self)
		diff[self.dirichlet_indices] = 0. # Do not modify constrained nodes
		ret[n*(order-1):] = diff
		return ret

	''' Convex stepping ''' 

	def step_cvx(self, dt: float):
		# Update boundary conditions
		self._t += dt
		self._t_prb.value = self._t
		self.update_constraints(self._t)
		self._y_prb.value = self._y
		self._prb.solve(warm_start=True)
		assert self._prb.status == 'optimal', f'CVXPY solve unsuccessful, status is: {self._prb.status}'
		self._y = self._prb.value
		self.set_constraints()

	''' Discrete stepping ''' 

	def step_map(self, dt: float):
		self._t += dt
		if (self._t - self._n) >= 1.0:
			self._n += 1
			self.update_constraints(self._t)
			self._y = self.map_fun(self._y) 
			self.set_constraints()
			self._y[self.dirichlet_indices] = self.dirichlet_values

	''' Constaint setting '''

	def update_constraints(self, t: float):
		''' Update the possibly time-varying state constraints ''' 
		if self.dynamic_dirichlet:
			self.dirichlet_values = np.array([self.dirichlet_fun(t, x) for x in self.X_dirichlet])
			if self.iter_mode is IterationMode.cvx:
				self._y_cstr.value = self.dirichlet_values
		if self.dynamic_neumann:
			self.neumann_values = np.array([self.neumann_fun(t, x) for x in self.X_neumann])

	def set_constraints(self):
		''' Set the state constraints ''' 
		if self.iter_mode is IterationMode.dydt:
			self.integrator.y[self.dirichlet_indices] = self.dirichlet_values
			self.integrator.y = self.project_fun(self.integrator.y)
		elif self.iter_mode is IterationMode.cvx:
			# No need to set boundary conditions since guaranteed by solution
			self._y = self.project_fun(self._y)
		elif self.iter_mode is IterationMode.map:
			self._y[self.dirichlet_indices] = self.dirichlet_values
			self._y = self.project_fun(self._y)

	''' Properties ''' 

	@property
	def y(self):
		if self.iter_mode is IterationMode.dydt:
			return self.integrator.y[:self.ndim]
		elif self.iter_mode is IterationMode.cvx or self.iter_mode is IterationMode.map:
			return self._y

	@property
	def t(self):
		if self.iter_mode is IterationMode.dydt:
			return self.integrator.t
		elif self.iter_mode is IterationMode.cvx or self.iter_mode is IterationMode.map:
			return self._t

	@property 
	def dt(self):
		assert self.iter_mode is IterationMode.dydt, 'Can only get step size of differential stepper'
		return self.max_step if self.stepper.step_size is None else self.stepper.step_size

	def system(self, name: str) -> System:
		return System(self, {name: self})

''' PDE on graph domain ''' 

class GraphObservable(Observable):
	''' Graph-domain observable ''' 
	def __init__(self, G: nx.Graph, Gd: GraphDomain):
		self.G = G
		self.Gd = Gd
		# Domains
		self.nodes = {v: i for i, v in enumerate(G.nodes())}
		self.nodes_i = {i: v for v, i in self.nodes.items()}
		self.edges = bidict({e: i for i, e in enumerate(G.edges())})
		self.edges_i = {i: e for e, i in self.edges.items()}
		self.triangles, tri_index = {}, 0
		for clique in nx.find_cliques(G):
			if len(clique) == 3:
				self.triangles[tuple(clique)] = tri_index
				tri_index += 1

		if Gd is GraphDomain.nodes:
			X = self.nodes
		elif Gd is GraphDomain.edges:
			X = self.edges
		elif Gd is GraphDomain.triangles:
			X = self.triangles
		Observable.__init__(self, X)

	def project(self, Gd: GraphDomain, view: Callable[['GraphObservable'], np.ndarray]) -> 'GraphObservable':
		class ProjectedObservable(GraphObservable):
			@property
			def y(other):
				return view(self)
			@property
			def t(other):
				return self.t
		return ProjectedObservable(self.G, Gd)

class gpde(pde, GraphObservable):
	def __init__(self, G: nx.Graph, Gd: GraphDomain, *args, w_key: str=None, **kwargs):
		GraphObservable.__init__(self, G, Gd)

		# Weights
		self.weights = np.ones(len(G.edges()))
		if w_key is not None:
			for i, e in enumerate(G.edges()):
				self.weights[i] = G[e[0]][e[1]][w_key]

		# Orientation / incidence
		self.orientation = {**{e: 1 for e in self.edges}, **{(e[1], e[0]): -1 for e in self.edges}} # Orientation implicit by stored keys in domain
		self.incidence = nx.incidence_matrix(G, oriented=True).multiply(np.sqrt(self.weights)) # |V| x |E| incidence

		# Operators
		self.vertex_laplacian = -self.incidence@self.incidence.T # |V| x |V| laplacian operator
		self.edge_laplacian = -self.incidence.T@self.incidence # |E| x |E| laplacian operator
		def curl_element(tri, edge):
			if edge[0] in tri and edge[1] in tri:
				c = np.sqrt(self.weights[self.edges[edge]])
				if edge == (tri[0], tri[1]) or edge == (tri[1], tri[2]) or edge == (tri[2], tri[0]): # Orientation of triangle
					return c
				return -c
			return 0
		self.curl3 = sparse_product(self.triangles.keys(), self.edges.keys(), curl_element) # |T| x |E| curl operator, where T is the set of 3-cliques in G; respects implicit orientation

		pde.__init__(self, self.X, *args, **kwargs)

	def set_boundary(self, 
			dirichlet: Callable[[Time, Point], float]=None, 
			neumann: Callable[[Time, Point], float]=None,
			dynamic: bool=False
		):
		pde.set_boundary(self, dirichlet, neumann, dynamic)
		if self.Gd is GraphDomain.nodes:
			self.dirichlet_laplacian = self.vertex_laplacian.copy()
			self.dirichlet_laplacian[self.dirichlet_indices, :] = 0
			self.dirichlet_laplacian.eliminate_zeros()
			self.neumann_correction = np.zeros_like(self.y)
			self.neumann_correction[self.neumann_indices] = self.neumann_values
		else:
			self.dirichlet_laplacian = self.edge_laplacian.copy()
			self.dirichlet_laplacian[self.dirichlet_indices, :] = 0
			self.dirichlet_laplacian.eliminate_zeros()
			# TODO: neumann conditions