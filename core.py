''' Specifying partial differential equations on graph domains ''' 

import numpy as np
import networkx as nx
from typing import Any, Union, Tuple, Callable, Newtype, Iterable
from scipy.integrate import RK45
from abc import ABC, abstractmethod

''' Common types ''' 

Vertex = Any
Edge = Tuple[Vertex, Vertex]
Face = Tuple[Vertex, ...]
Point = Union[Vertex, Edge, Face] # A point in the graph domain
Time = Newtype('Time', float)
Domain = Dict[Point, int] # Mapping from points into array indices

''' Base class ''' 

class gpde(ABC):
	def __init__(self, X: Domain, f: Callable[[Time], np.ndarray], order: int=1, max_step=None):
		''' Create a PDE defined on some domain.
		Uses RK45 solver.
		''' 
		self.X = X
		self.Xi = list(X.values())
		self.ndim = len(X)
		self.t0 = 0.
		self.y0 = np.full(len(X), np.nan)
		self.f = f
		self.order = order
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=max_step)
		self.w_key = w_key
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

	def __getitem__(self, idx):
		return self.integrator.y.__getitem__(idx)

''' Derived classes ''' 

class vertex_pde(gpde):
	''' PDE defined on the vertices of a graph ''' 
	def __init__(self, G: nx.Graph, *args, w_key: str=None, **kwargs):
		X = {x: i for i, x in enumerate(G.nodes())}
		self.X_from = [e[0] for e in G.edges()]
		self.X_to = [e[1] for e in G.edges()]
		self.G = G
		self.laplacian_matrix = -nx.laplacian_matrix(G)
		self.weights = np.ones(len(G.edges()))
		if w_key is not None:
			for i, e in enumerate(G.edges()):
				self.weights[i] = G[e[0]][e[1]][w_key]
		super().__init__(X, *args, **kwargs)

	''' Available spatial differential operators '''

	def grad(self) -> np.ndarray:
		return np.sqrt(self.weights) * (self.y[self.X_from] - self.y[self.X_to])

	def laplacian(self) -> np.ndarray:
		fixed_flux = replace(np.zeros(self.y.shape), self.neumann_indices, [self.neumann(self.t, x) for x in self.neumann_X])
		return self.laplacian_matrix@self.y + fixed_flux

	def bilaplacian(self) -> np.ndarray:
		# TODO correct way to handle Neumann in this case? (Gradient constraint only specifies one neighbor beyond)
		return self.laplacian_matrix@self.laplacian()

	def advect(self, v_field: Callable[[Edge], float]) -> np.ndarray:
		pass


class edge_pde(gpde):
	''' PDE defined on the edges of a graph ''' 
	pass

class face_pde(gpde):
	''' PDE defined on the faces of a graph ''' 
	pass

