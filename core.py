''' Specifying partial differential equations on graph domains ''' 

import numpy as np
import networkx as nx
import networkx.linalg.graphmatrix as nx
from typing import Any, Union, Tuple, Callable, Newtype, Iterable
from scipy.integrate import RK45
from abc import ABC, abstractmethod

from .utils import *

''' Common types ''' 

Vertex = Any
Edge = Tuple[Vertex, Vertex]
Face = Tuple[Vertex, ...]
Point = Union[Vertex, Edge, Face] # A point in the graph domain
Time = Newtype('Time', float)
Domain = Dict[Point, int] # Mapping from points into array indices
Sign = NewType('Sign', int)

''' Base interface ''' 

class Integrable(ABC):
	@abstractmethod
	def step(self, dt: float):
		pass

	@abstractmethod
	def reset(self):
		pass

class Observable(ABC):
	@property
	@abstractmethod
	def t(self): np.ndarray:
		pass

	@property
	@abstractmethod
	def y(self): np.ndarray:
		pass

	def __getitem__(self, idx):
		return self.y.__getitem__(idx)

''' Base classes ''' 

class pde(Observable, Integrable):
	def __init__(self, X: Domain, f: Callable[[Time], np.ndarray], order: int=1, max_step=None):
		''' Create a PDE defined on some domain.
		Uses RK45 solver.
		''' 
		self.X = X
		self.Xi = list(X.values())
		self.ndim = len(X)
		self.t0 = 0.
		self.y0 = np.full(self.ndim*order, np.nan)
		self.f = f
		self.order = order
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=max_step)
		self.dirichlet = lambda t, x: None
		self.dirichlet_X = set()
		self.dirichlet_indices = []
		self.neumann = lambda t, x: None
		self.neumann_X = set()
		self.neumann_indices = []
		self.nonphysical = lambda y: False 

	def set_initial(self, t0: float=0., y0: Callable[[Point], float]=lambda _: 0., **kwargs):
		''' Set initial values. Other optional arguments are taken to be nth-derivative initial values. ''' 
		assert len(kwargs) == self.order - 1, f'{len(kwargs)+1} initial conditions provided but {self.order} needed'
		self.t0 = t0
		self.integrator.t = t0
		for x, i in self.X:
			self.y0[i] = y0(x)
			self.integrator.y[i] = y0(x)
			for j, y0_j in enumerate(kwargs):
				self.y0[(j+1)*ndim + i] = y0_i(x)
				self.integrator.y[(j+1)*ndim + i] = y0_i(x)

	def set_boundary(self, 
			dirichlet: Callable[[Time, Point], float]=lambda t, x: None, 
			neumann: Callable[[Time, Point], float]=lambda t, x: None
		):
		''' Impose boundary conditions via time-varying values (dirichlet conditions) or fluxes (neumann conditions). 
		Assumes the domains do not change. 
		''' 
		self.dirichlet = dirichlet
		self.dirichlet_X = set(x if dirichlet(x) is not None for x in self.X) # Fixed-value domain
		self.dirichlet_indices = [self.X[x] for x in self.dirichlet_X]
		self.neumann = neumann
		self.neumann_X = set(x if neumann(x) is not None for x in self.X) # Fixed-flux domain
		self.neumann_indices = [self.X[x] for x in self.neumann_X]
		intersect = self.dirichlet_X & self.neumann_X
		assert len(intersect) == 0, f'Dirichlet and Neumann conditions overlap on {intersect}'


	def step(self, dt: float):
		''' Integrate forward in time ''' 
		self.integrator.t_bound = self.t + dt
		self.integrator.status = 'running'
		while self.integrator.status != 'finished':
			self.integrator.step()
			for x in self.dirichlet_X:
				self.integrator.y[self.X[x] - self.ndim] = self.dirichlet(self.integrator.t, x)

	def reset(self):
		''' Reset the system to initial conditions ''' 
		pass

	def __call__(self, x: Point):
		''' Evaluate at a point '''
		return self.integrator.y[self.X[x]]

	''' Private methods ''' 

	def dydt(self, t: Time, y: np.ndarray):
		n, order = self.ndim, self.order
		ret = np.zeros_like(y)
		for i in range(order-1):
			ret[n*i:n*(i+1)] = y[n*(i+1):n*(i+2)]
		diff = self.f(t, y[:n])
		diff = replace(diff, self.dirichlet_indices, np.zeros(len(self.dirichlet_X))) # Do not modify constrained nodes
		ret[n*(order-1):] = diff
		return ret

	@property
	def y(self):
		return self.integrator.y[:self.ndim]

	@property
	def t(self):
		return self.integrator.t

class multi_pde(Integrable):
	''' Multiple PDEs coupled in time. Can be integrated but not observed directly. ''' 
	def __init__(self, *pdes: Tuple[pde], max_step=None):
		assert len(pdes) >= 1 
		assert all([p.t == pdes[0].t for p in pdes]), 'Cannot couple integrators at different times'
		self.pdes = pdes
		self.max_step = max_step
		self.t0 = pdes[0].t
		y0s = [p.integrator.y for p in pdes]
		self.views = [slice(0, len(y0s[0]))]
		for i in range(1, len(pdes)):
			start = self.views[i-1].stop
			self.views.append(slice(start, start+len(y0s[i])))
		self.y0 = np.concatenate(y0s)
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=max_step)

	def dydt(self, t: Time, y: np.ndarray):
		return np.concatenate([p.dydt(t, y) for p in self.pdes])

	def step(self, dt: float):
		self.integrator.t_bound = self.t + dt
		self.integrator.status = 'running'
		while self.integrator.status != 'finished':
			self.integrator.step()
			# Apply all boundary conditions
			for p, view in zip(self.xs, self.views):
				for x in p.dirichet_X:
					self.integrator.y[view][p.X[x] - p.ndim] = p.dirichlet(self.integrator.t, x)

	def observables(self) -> List[Observable]:
		class ViewObservable(Observable):
			def __init__(obs, view: slice):
				obs.view = view
			@property
			def t(obs):
				return self.integrator.t
			@property
			def y(obs):
				return self.integrator.y[obs.view]
		return [ViewObservable(view) for view in self.views]

	def reset(self):
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=self.max_step)


''' Derived classes ''' 

class vertex_pde(pde):
	''' PDE defined on the vertices of a graph ''' 
	def __init__(self, G: nx.Graph, *args, w_key: str=None, **kwargs):
		X = {x: i for i, x in enumerate(G.nodes())}
		self.edge_X = {x: i for i, x in enumerate(G.nodes())}
		self.X_from = [e[0] for e in G.edges()]
		self.X_to = [e[1] for e in G.edges()]
		self.G = G
		self.laplacian_matrix = -nx.laplacian_matrix(G)
		self.weights = np.ones(len(G.edges()))
		if w_key is not None:
			for i, e in enumerate(G.edges()):
				self.weights[i] = G[e[0]][e[1]][w_key]
		super().__init__(X, *args, **kwargs)

	''' Spatial differential operators '''

	def partial(self, e: Edge) -> float:
		return np.sqrt(self.weights[self.edge_X[e]]) * (self(e[1]) - self(e[0]))

	def grad(self) -> np.ndarray:
		return np.sqrt(self.weights) * (self.y[self.X_from] - self.y[self.X_to])

	def laplacian(self) -> np.ndarray:
		fixed_flux = replace(np.zeros(self.y.shape), self.neumann_indices, [self.neumann(self.t, x) for x in self.neumann_X])
		return self.laplacian_matrix@self.y + fixed_flux

	def bilaplacian(self) -> np.ndarray:
		# TODO correct way to handle Neumann in this case? (Gradient constraint only specifies one neighbor beyond)
		return self.laplacian_matrix@self.laplacian()

	def advect(self, v_field: Callable[[Edge], float]) -> np.ndarray:
		return np.array([
			sum([v_field((x, y)) * self.partial((x, y)) for y in self.neighbors(x)])
			for x in self.X
		])


class edge_pde(pde):
	''' PDE defined on the edges of a graph ''' 
	def __init__(self, G: nx.Graph, *args, w_key: str=None, orientation: Callable[[Edge], Sign]=None, **kwargs):
		X = bidict({x: i for i, x in enumerate(G.edges())})
		self.G = G
		self.weights = np.ones(len(G.edges()))
		if w_key is not None:
			for i, e in enumerate(G.edges()):
				self.weights[i] = G[e[0]][e[1]][w_key]
		self.incidence = nx.incidence_matrix(G)
		# Orient edges
		self.orientation = np.ones(len(X)) if orientation is None else np.array([orientation(x) for x in X])
		self.oriented_incidence = self.incidence.copy()
		for i, x in enumerate(G.edges()):
			if self.orientation[i] > 0:
				self.oriented_incidence[x[0]][i] = -1
			else:
				self.oriented_incidence[x[1]][i] = -1
		# Vertex dual
		self.G_dual = nx.line_graph(G)
		self.X_dual = bidict({x: i for i, x in enumerate(self.G_dual.edges())})
		self.weights_dual = np.zeros(len(self.X_dual))
		for i, x in enumerate(self.G_dual.edges()):
			for n in destructure(x):
				if G.degree[n] > 1:
					(a, b) = x
					self.weights_dual[self.X_dual[x]] += 
						self.incidence[n][a] * self.weights[X[a]] * 
						self.incidence[n][b] * self.weights[X[b]] / 
						(G.degree[n] - 1)
		# Orientation of dual
		self.oriented_incidence_dual = nx.incidence_matrix(self.G_dual)
		for i, x in enumerate(self.X):
			for y in self.G_dual.neighbors(x):
				if y[1] in x: # Inward edges receive negative orientation
					j = self.X_dual[(x,y)]
					self.oriented_incidence_dual[i][j] = -1
		super().__init__(X, *args, **kwargs)

	def __call__(self, x: Edge):
		return self.orientation[self.X[x]] * self.integrator.y[self.X[x]]

	''' Spatial differential operators '''

	def div(self) -> np.ndarray:
		return 2 * np.sqrt(self.weights) * self.y@self.incidence.T

	def curl(self) -> np.ndarray:
		raise NotImplementedError

	def laplacian(self) -> np.ndarray:
		''' Vector laplacian https://en.wikipedia.org/wiki/Vector_Laplacian ''' 
		x1 = np.sqrt(self.weights) * self.div()@self.oriented_incidence
		x2 = curl_operator@self.curl() # TODO		
		return x1 - x2

	def advect(self, v_field: Callable[[Edge], float]) -> np.ndarray:
		ret = np.zeros(self.ndim)
		for a, i in self.X.items():
			u = self(a)
			for b in self.G_dual.neighbors(a):
				j = self.X_dual[(a, b)]
				ret[i] += self.oriented_incidence_dual[i][j] * v_field(b) * u / self.weights_dual[j]
		return np.array(ret)

	''' Private methods ''' 

	@property
	def y(self) -> np.ndarray:
		return self.integrator.y[:self.ndim] * self.orientation

class face_pde(gpde):
	''' PDE defined on the faces of a graph ''' 
	def __init__(self, G: nx.Graph, *args, **kwargs):
		# Let faces be represented by cycles
		X = {x: i for i, x in enumerate(nx.cycle_basis(G))}
		self.G = G
		super().__init__(X, *args, **kwargs)

