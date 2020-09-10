''' Specifying partial differential equations on graph domains ''' 

import numpy as np
import networkx as nx
# import networkx.linalg.graphmatrix as nx
from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict
from scipy.integrate import RK45
from abc import ABC, abstractmethod
import pdb

from .utils import *

''' Common types ''' 

Time = NewType('Time', float)
Point = Any
Domain = Dict[Point, int] # Mapping from points into array indices
Sign = NewType('Sign', int)
System = Tuple['Integrable', List['Observable']]

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
	def t(self) -> np.ndarray:
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

''' Base class: PDE on arbitrary domain ''' 

class pde(Observable, Integrable):
	def __init__(self, X: Domain, f: Callable[[Time, 'pde'], np.ndarray], order: int=1, max_step=1e-3):
		''' Create a PDE defined on some domain.
		Uses RK45 solver.
		''' 
		Observable.__init__(self, X)
		self.Xi = list(X.values())
		self.t0 = 0.
		self.y0 = np.zeros(self.ndim*order)
		self.f = f
		self.max_step = max_step
		self.order = order
		self.integrator = RK45(lambda t, y: np.zeros_like(self.y0), self.t0, self.y0, np.inf, max_step=max_step)
		self.integrator.fun = self.dydt
		self.dynamic_bc = True
		self.dirichlet = lambda t, x: None
		self.dirichlet_X = set()
		self.dirichlet_indices = []
		self.neumann = lambda t, x: None
		self.neumann_X = set()
		self.neumann_indices = []
		self.neumann_values = []
		self.erroneous = lambda y: False 

	def set_initial(self, t0: float=0., y0: Callable[[Point], float]=lambda _: 0., **kwargs):
		''' Set initial values. Other optional arguments are taken to be nth-derivative initial values. ''' 
		assert len(kwargs) == self.order - 1, f'{len(kwargs)+1} initial conditions provided but {self.order} needed'
		self.t0 = t0
		self.integrator.t = t0
		for x, i in self.X.items():
			self.y0[i] = y0(x)
			self.integrator.y[i] = y0(x)
			for j, y0_j in enumerate(kwargs):
				self.y0[(j+1)*ndim + i] = y0_i(x)
				self.integrator.y[(j+1)*ndim + i] = y0_i(x)

	def set_boundary(self, 
			dirichlet: Callable[[Time, Point], float]=lambda t, x: None, 
			neumann: Callable[[Time, Point], float]=lambda t, x: None,
			dynamic: bool=True
		):
		''' Impose boundary conditions via time-varying values (dirichlet conditions) or fluxes (neumann conditions). 
		Assumes the domains do not change. If `dynamic` is off, the boundary conditions are assumed to be static for performance improvement.
		''' 
		self.dynamic_bc = dynamic
		self.dirichlet = dirichlet
		self.dirichlet_X = set({x for x in self.X if (dirichlet(0., x) is not None)}) # Fixed-value domain
		for x in self.dirichlet_X:
			v, i = self.dirichlet(0., x), self.X[x] - self.ndim
			self.integrator.y[i] = self.y0[i] = v 
		self.dirichlet_indices = [self.X[x] for x in self.dirichlet_X]
		self.neumann = neumann
		self.neumann_X = set({x for x in self.X if (neumann(0., x) is not None)}) # Fixed-flux domain
		self.neumann_indices = [self.X[x] for x in self.neumann_X]
		self.neumann_values = np.array([neumann(0., x) for x in self.neumann_X])
		intersect = self.dirichlet_X & self.neumann_X
		assert len(intersect) == 0, f'Dirichlet and Neumann conditions overlap on {intersect}'

	def step(self, dt: float):
		''' Integrate forward in time ''' 
		self.integrator.t_bound = self.t + dt
		self.integrator.status = 'running'
		while self.integrator.status != 'finished':
			self.integrator.step()
			# Update boundary conditions
			if self.dynamic_bc:
				for x in self.dirichlet_X:
					self.integrator.y[self.X[x] - self.ndim] = self.dirichlet(self.integrator.t, x)
		if self.erroneous(self.y):
			raise ValueError('Erroneous state encountered')

	def reset(self):
		''' Reset the system to initial conditions ''' 
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=self.max_step)

	def set_erroneous(self, erroneous: Callable[[np.ndarray], bool]):
		self.erroneous = erroneous

	def system(self) -> System:
		''' Express as a single-observable system ''' 
		return (self, [self]) 

	''' Private methods ''' 

	def dydt(self, t: Time, y: np.ndarray):
		n, order = self.ndim, self.order
		ret = np.zeros_like(y)
		for i in range(order-1):
			ret[n*i:n*(i+1)] = y[n*(i+1):n*(i+2)]
		# Update boundary conditions
		if self.dynamic_bc:
			self.neumann_values = np.array([self.neumann(t, x) for x in self.neumann_X])
		diff = self.f(t, self)
		diff = replace(diff, self.dirichlet_indices, np.zeros(len(self.dirichlet_X))) # Do not modify constrained nodes
		ret[n*(order-1):] = diff
		return ret

	@property
	def y(self):
		return self.integrator.y[:self.ndim]

	@property
	def t(self):
		return self.integrator.t

''' PDE on graph domain ''' 

Vertex = Any
Edge = Tuple[Vertex, Vertex]
Face = Tuple[Vertex, ...]
Point = Union[Vertex, Edge, Face] # A point in the graph domain

class gpde(pde):
	def __init__(self, G: nx.Graph, *args, w_key: str=None, **kwargs):
		self.G = G
		# Domains
		self.vertices = {v: i for i, v in enumerate(G.nodes())}
		self.edges = bidict({e: i for i, e in enumerate(G.edges())})
		self.triangles, tri_index = {}, 0
		for clique in nx.find_cliques(G):
			if len(clique) == 3:
				self.triangles[tuple(clique)] = tri_index
				tri_index += 1

		# Weights
		self.weights = np.ones(len(G.edges()))
		if w_key is not None:
			for i, e in enumerate(G.edges()):
				self.weights[i] = G[e[0]][e[1]][w_key]

		# Orientation / incidence
		self.orientation = {**{e: 1 for e in self.edges}, **{(e[1], e[0]): -1 for e in self.edges}} # Orientation implicit by stored keys in domain
		self.incidence = nx.incidence_matrix(G, oriented=True)

		# Operators
		self.vertex_laplacian = -nx.laplacian_matrix(G) # |V| x |V| laplacian operator
		self.gradient = self.incidence.multiply(np.sqrt(self.weights)).T # |E| x |V| gradient operator; respects implicit orientation
		self.curl3 = sparse_product(
			self.triangles.keys(), 
			self.edges.keys(), 
			lambda t, e: float(e[0] in t and e[1] in t) / np.sqrt(self.weights[self.edges[e]])
		) # |T| x |E| curl operator, where T is the set of 3-cliques in G; respects implicit orientation

		super().__init__(self.get_domain(), *args, **kwargs)


	@abstractmethod
	def get_domain(self) -> Domain:
		''' Subclasses must specify the domain (vertices, edges, etc.) '''
		pass