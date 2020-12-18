''' Specifying partial differential equations on graph domains ''' 

import numpy as np
import networkx as nx
# import networkx.linalg.graphmatrix as nx
from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict
from scipy.integrate import RK45
from scipy.optimize import least_squares
from abc import ABC, abstractmethod
import pdb
from enum import Enum
import os.path
import os
import hickle as hkl
import cloudpickle
from tqdm import tqdm

from .utils import *

''' Common types ''' 

Time = NewType('Time', float)
Point = Any
Domain = Dict[Point, int] # Mapping from points into array indices
Sign = NewType('Sign', int)

class SolveMode(Enum):
	forward = 0
	direct = 1

''' Base interfaces ''' 

class Integrable(ABC):
	@abstractmethod
	def step(self, dt: float):
		pass

	@abstractmethod
	def reset(self):
		pass

class Observable(ABC):
	def __init__(self, X: Domain):
		self.X = X # Domain
		self.ndim = len(X)

	@property
	@abstractmethod
	def t(self) -> float:
		pass

	@property
	@abstractmethod
	def y(self) -> np.ndarray:
		pass

	def __getitem__(self, idx):
		return self.y.__getitem__(idx)

	def __call__(self, x: Point):
		''' Measure at a point '''
		return self.y[self.X[x]]

	def __len__(self):
		return self.ndim

''' System objects ''' 

class System:
	def __init__(self, integrator: Integrable, observables: Dict[str, Observable]):
		self._integrator = integrator
		self._observables = observables

	@property
	def integrator(self):
		return self._integrator

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
					self.integrator.step(dt)
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

		class DummyIntegrable(Integrable):
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


''' Base class: PDE on arbitrary domain ''' 

class pde(Observable, Integrable):
	def __init__(self, 
			X: Domain, 
			dydt: Callable[[Time, 'pde'], np.ndarray] = None, 
			lhs: Callable[[Time, 'pde'], np.ndarray] = None,
			order: int=1, max_step=1e-3, gtol=1e-3
		):
		''' Create a PDE defined on some domain.
		Args:
			X: solution domain
		(either one of the following must be provided)
			dydt: time-difference of solution [integrated with RK45]
			lhs: left-hand side of zero-valued equation [solved directly with least-squares]
		Optional args:
			order: (forward) order of dydt
			max_step: (forward) max step for RK solver
			gtol: (direct) error tolerance for least-squares
		''' 
		assert (dydt is not None or lhs is not None), 'Either pass a time-difference or LHS of equation to be satisfied'
		Observable.__init__(self, X)
		self.Xi = list(X.values())
		self.t0 = 0.
		self.dynamic_bc = True
		self.dirichlet = lambda t, x: None
		self.dirichlet_X = set()
		self.dirichlet_indices = []
		self.neumann = lambda t, x: None
		self.neumann_X = set()
		self.neumann_indices = []
		self.neumann_values = []
		self.erroneous = None

		if dydt is not None:
			self.mode = SolveMode.forward
			self.y0 = np.zeros(self.ndim*order)
			self.dydt_fun = dydt
			self.max_step = max_step
			self.order = order
			self.integrator = RK45(lambda t, y: np.zeros_like(self.y0), self.t0, self.y0, np.inf, max_step=max_step)
			self.integrator.fun = self.dydt
		else:
			self.t_direct = self.t0
			self.mode = SolveMode.direct
			self.y0 = np.zeros(self.ndim)
			self.lhs_fun = lhs
			self.y_direct = np.zeros_like(self.y0)
			self.gtol = gtol

	def set_initial(self, t0: float=0., y0: Callable[[Point], float]=lambda _: 0., **kwargs):
		''' Set initial values. Other optional arguments are taken to be nth-derivative initial values. ''' 
		self.t0 = t0
		if self.mode is SolveMode.forward:
			assert len(kwargs) == self.order - 1, f'{len(kwargs)+1} initial conditions provided but {self.order} needed'
			self.integrator.t = t0
		for x in self.X.keys() - self.dirichlet_X:
			i = self.X[x]
			self.y0[i] = y0(x)
			if self.mode is SolveMode.forward:
				self.integrator.y[i] = self.y0[i]
				for j, y0_j in enumerate(kwargs):
					self.y0[(j+1)*ndim + i] = y0_i(x)
					self.integrator.y[(j+1)*ndim + i] = y0_i(x)
			else:
				self.y_direct[i] = self.y0[i]

	def set_boundary(self, 
			dirichlet: Callable[[Time, Point], float]=lambda x: None, 
			neumann: Callable[[Time, Point], float]=lambda x: None,
			dynamic: bool=False
		):
		''' Impose boundary conditions via time-varying values (dirichlet conditions) or fluxes (neumann conditions). 
		Assumes the domains do not change. If `dynamic` is off, the boundary conditions are assumed to be static for performance improvement.
		''' 
		self.dynamic_bc = dynamic
		self.dirichlet = dirichlet
		self.dirichlet_X = set({x for x in self.X if ((dirichlet(0., x) if dynamic else dirichlet(x)) is not None)}) # Fixed-value domain
		self.dirichlet_indices = np.array([self.X[x] for x in self.dirichlet_X], dtype=np.int64)
		self.dirichlet_values = np.array([(dirichlet(0., x) if dynamic else dirichlet(x)) for x in self.dirichlet_X])
		self.y0 = replace(self.y0, self.dirichlet_indices, self.dirichlet_values)
		if self.mode is SolveMode.forward:
			for i, v in zip(self.dirichlet_indices, self.dirichlet_values):
				self.integrator.y[i - self.ndim] = v 
		else:
			self.y_direct = self.y0.copy()
		self.neumann = neumann
		self.neumann_X = set({x for x in self.X if ((neumann(0., x) if dynamic else neumann(x)) is not None)}) # Fixed-flux domain
		self.neumann_indices = np.array([self.X[x] for x in self.neumann_X], dtype=np.int64)
		self.neumann_values = np.array([(neumann(0., x) if dynamic else neumann(x)) for x in self.neumann_X])
		intersect = self.dirichlet_X & self.neumann_X
		assert len(intersect) == 0, f'Dirichlet and Neumann conditions overlap on {intersect}'

	def step(self, dt: float):
		''' Solve system at t+dt ''' 
		if self.mode is SolveMode.forward:
			self.step_forward(dt)
		else:
			self.step_direct(dt)

	def reset(self):
		''' Reset the system to initial conditions ''' 
		if self.mode is SolveMode.forward:
			self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=self.max_step)
		else:
			self.t_direct = self.t0
			self.y_direct = self.y0.copy()

	def set_erroneous(self, erroneous: Callable[[np.ndarray], bool]):
		self.erroneous = erroneous

	''' Solving / private methods ''' 

	def step_forward(self, dt: float):
		''' Integrate forward in time ''' 
		self.integrator.t_bound = self.t + dt
		self.integrator.status = 'running'
		while self.integrator.status != 'finished':
			self.integrator.step()
			if self.dynamic_bc:
				self.integrator.y[self.dirichlet_indices - ndim] = self.dirichlet_values
		if self.erroneous is not None and self.erroneous(self.y):
			raise ValueError('Erroneous state encountered')

	def dydt(self, t: Time, y: np.ndarray):
		n, order = self.ndim, self.order
		ret = np.zeros_like(y)
		for i in range(order-1):
			ret[n*i:n*(i+1)] = y[n*(i+1):n*(i+2)]
		# Update boundary conditions
		if self.dynamic_bc:
			self.update_bcs(t)
		diff = self.dydt_fun(t, self)
		diff[self.dirichlet_indices] = 0. # Do not modify constrained nodes
		ret[n*(order-1):] = diff
		return ret

	def step_direct(self, dt: float):
		# Update boundary conditions
		self.t_direct += dt
		if self.dynamic_bc:
			self.update_bcs(self.t_direct)
		# TODO: Jacobian?
		result = least_squares(self.lhs, self.y_direct, gtol=self.gtol)
		if result.status >= 1:
			self.y_direct = result.x
		else:
			raise Exception(f'Direct solver failed: {result.message}')

	def lhs(self, y: np.ndarray):
		self.y_direct = replace(y, self.dirichlet_indices, self.dirichlet_values) # Do not modify constrained nodes
		return self.lhs_fun(self.t, self)

	def update_bcs(self, t: float):
		self.neumann_values = np.array([self.neumann(t, x) for x in self.neumann_X])
		self.dirichlet_values = np.array([self.dirichlet(t, x) for x in self.dirichlet_X])

	@property
	def y(self):
		if self.mode is SolveMode.forward:
			return self.integrator.y[:self.ndim]
		else:
			return self.y_direct

	@property
	def t(self):
		if self.mode is SolveMode.forward:
			return self.integrator.t
		else:
			return self.t_direct

''' PDE on graph domain ''' 

Vertex = Any
Edge = Tuple[Vertex, Vertex]
Triangle = Tuple[Vertex, Vertex, Vertex]
Point = Union[Vertex, Edge, Triangle] # A point in the graph domain
class GraphDomain(Enum):
	vertices = 0
	edges = 1
	triangles = 2

class GraphObservable(Observable):
	''' Graph-domain observable ''' 
	def __init__(self, G: nx.Graph, Gd: GraphDomain):
		self.G = G
		self.Gd = Gd
		# Domains
		self.vertices = {v: i for i, v in enumerate(G.nodes())}
		self.edges = bidict({e: i for i, e in enumerate(G.edges())})
		self.triangles, tri_index = {}, 0
		for clique in nx.find_cliques(G):
			if len(clique) == 3:
				self.triangles[tuple(clique)] = tri_index
				tri_index += 1

		if Gd is GraphDomain.vertices:
			X = self.vertices
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
		self.incidence = nx.incidence_matrix(G, oriented=True).multiply(np.sqrt(self.weights))

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
