import numpy as np 
from typing import Tuple
from scipy.integrate import RK45

from .core import *
from .utils import *

''' Derived interfaces ''' 

class VertexObservable(Observable):
	def __init__(self, G: nx.Graph):
		self.G = G
		X = bidict({x: i for i, x in enumerate(G.nodes())})
		super().__init__(X)

class EdgeObservable(Observable):
	def __init__(self, G: nx.Graph):
		self.G = G
		X = bidict({x: i for i, x in enumerate(G.edges())})
		super().__init__(X)

class FaceObservable(Observable):
	def __init__(self, G: nx.Graph):
		self.G = G
		# Let faces be represented by cycles
		X = {x: i for i, x in enumerate(nx.cycle_basis(G))}
		super().__init__(X)

''' Derived PDE types ''' 

class vertex_pde(pde, VertexObservable):
	''' PDE defined on the vertices of a graph ''' 
	def __init__(self, G: nx.Graph, *args, w_key: str=None, **kwargs):
		VertexObservable.__init__(self, G)
		self.edge_X = {x: i for i, x in enumerate(G.nodes())}
		self.X_from = [e[0] for e in G.edges()]
		self.X_to = [e[1] for e in G.edges()]
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


class edge_pde(pde, EdgeObservable):
	''' PDE defined on the edges of a graph ''' 
	def __init__(self, G: nx.Graph, *args, w_key: str=None, orientation: Callable[[Edge], Sign]=None, **kwargs):
		EdgeObservable.__init__(self, G)
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
		return self.orientation[self.X[x]] * self.y[self.X[x]]

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

class face_pde(pde, FaceObservable):
	''' PDE defined on the faces of a graph ''' 
	pass		