import numpy as np
import networkx as nx
from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict

from .types import *
from .utils import *
from .fds import *

''' Observables on graph domains ''' 

class GraphObservable(Observable):
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

''' Dynamical systems on generic graph domains ''' 

class gds(fds, GraphObservable):
	def __init__(self, G: nx.Graph, Gd: GraphDomain, w_key: str=None):
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

		fds.__init__(self, self.X)

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

''' Dynamical systems on specific graph domains ''' 

class vertex_gds(gds):
	''' PDE defined on the nodes of a graph ''' 

	def __init__(self, G: nx.Graph, *args, **kwargs):
		gds.__init__(self, G, GraphDomain.nodes, *args, **kwargs)

	''' Differential operators '''

	def partial(self, e: Edge) -> float:
		return np.sqrt(self.weights[self.edges[e]]) * (self(e[1]) - self(e[0])) 

	def grad(self) -> np.ndarray:
		return self.incidence.T@self.y

	def laplacian(self, y: np.ndarray=None) -> np.ndarray:
		''' Dirichlet-Neumann Laplacian. TODO: should minimize error from laplacian on interior? ''' 
		if y is None:
			y = self.y
		return self.dirichlet_laplacian@y + self.neumann_correction

	def bilaplacian(self) -> np.ndarray:
		# TODO correct way to handle Neumann in this case? (Gradient constraint only specifies one neighbor beyond)
		return self.dirichlet_laplacian@self.laplacian()

	def advect(self, v_field: Callable[[Edge], float]) -> np.ndarray:
		if isinstance(v_field, edge_gds) and v_field.G is self.G:
			M = self.incidence.multiply(v_field.y)
			N = M.copy().T
			N.data[N.data > 0] = 0.
			N.data *= -1
			ret = -M@N@self.y
			ret[self.dirichlet_indices] = 0.
			return ret
		else:
			# This option is broken; need to properly use orientation
			assert False
			return np.array([
				sum([v_field((x, y)) * self.partial((x, y)) for y in self.G.neighbors(x)])
				for x in self.X
			])

class edge_gds(gds):
	''' PDE defined on the edges of a graph ''' 
	def __init__(self, G: nx.Graph, *args, **kwargs):
		gds.__init__(self, G, GraphDomain.edges, *args, **kwargs)
		# Vertex dual
		self.G_dual = nx.line_graph(G)
		self.X_dual = bidict({x: i for i, x in enumerate(self.G_dual.edges())})
		# self.edge_laplacian = -nx.laplacian_matrix(self.G_dual) # WRONG
		# self.weights_dual = np.zeros(len(self.X_dual))
		# for i, x in enumerate(self.G_dual.edges()):
		# 	for n in destructure(x):
		# 		if G.degree[n] > 1:
		# 			(a, b) = x
		# 			self.weights_dual[self.X_dual[x]] += (
		# 				self.incidence[n][a] * self.weights[self.X[a]] * 
		# 				self.incidence[n][b] * self.weights[self.X[b]] / 
		# 				(G.degree[n] - 1)
		# 			)
		self.weights_dual = np.ones(len(self.X_dual)) # TODO
		# Orientation of dual
		# self.oriented_incidence_dual = nx.incidence_matrix(self.G_dual)
		# for i, x in enumerate(self.X):
		# 	for y in self.G_dual.neighbors(x):
		# 		if y[1] in x: # Inward edges receive negative orientation
		# 			j = self.X_dual[(x,y)]
		# 			self.oriented_incidence_dual[i, j] = -1
		def adj_dual(n, e): 
			# TODO: use edge weights
			if n == e:
				return 0
			elif n[1] == e[1] or n[0] == e[1]:
				return 1
			elif n[0] == e[0] or n[1] == e[0]:
				return -1
			return 0
		self.adj_dual = sparse_product(self.edges.keys(), self.edges.keys(), adj_dual) # |E| x |E| signed edge adjacency matrix

	def __call__(self, x: Edge):
		return self.orientation[x] * self.y[self.X[x]]

	''' Spatial differential operators '''

	def div(self) -> np.ndarray:
		return -self.incidence@self.y

	def influx(self) -> np.ndarray:
		''' In-flux through nodes ''' 
		f = self.incidence.multiply(self.y)
		f.data[f.data < 0] = 0.
		return f.sum(axis=1)

	def outflux(self) -> np.ndarray:
		''' Out-flux through nodes ''' 
		f = -self.incidence.multiply(self.y)
		f.data[f.data < 0] = 0.
		return f.sum(axis=1)

	def curl(self) -> np.ndarray:
		raise self.curl3@self.y

	def laplacian(self) -> np.ndarray:
		''' Vector laplacian or discrete Helmholtz operator 
		https://www.stat.uchicago.edu/~lekheng/work/psapm.pdf 
		TODO: neumann conditions
		''' 
		return self.dirichlet_laplacian@self.y 

	def bilaplacian(self) -> np.ndarray:
		return self.dirichlet_laplacian@self.laplacian()

	def advect(self, v_field: Callable[[Edge], float] = None) -> np.ndarray:
		if v_field is None:
			# Transpose of scalar advection case.
			# TODO: does this assume flow is irrotational?
			M = self.incidence.multiply(self.y).T
			N = M.copy().T
			M.data[M.data > 0] = 0.
			M.data *= -1
			ret = -M@N@self.y
			ret[self.dirichlet_indices] = 0.
			return ret
		elif isinstance(v_field, edge_gds) and v_field.G is self.G:
			# Since graphs are identical, orientation is implicitly respected
			M = self.incidence.multiply(v_field.y).T
			N = M.copy().T
			M.data[M.data > 0] = 0.
			M.data *= -1
			return -M@N@self.y
		else:
			# TODO: incorrect
			ret = np.zeros(self.ndim)
			for a, i in self.X.items():
				u = self(a)
				for b in self.G_dual.neighbors(a):
					w = self.weights_dual[self.X_dual[(a, b)]]
					if b[0] == a[1]: # Outgoing edge
						ret[i] += u * v_field(b) / w
					elif b[1] == a[1]: # Outgoing edge, reversed direction
						ret[i] -= u * v_field(b) / w
					elif b[0] == a[0]: # Ingoing edge, reversed directopm
						ret[i] += u * v_field(b) / w
					else: # Ingoing edge
						ret[i] -= u * v_field(b) / w
			return np.array(ret)

	def dual(self) -> GraphObservable:
		''' View the dual graph; TODO unify this interface across domains ''' 
		G_ = nx.line_graph(self.G)
		edge_map = np.array([self.X[e] for e in G_.nodes], dtype=np.intp)
		class DualGraphObservable(GraphObservable):
			@property
			def y(other):
				return self.y[edge_map]
			@property
			def t(other):
				return self.t
		return DualGraphObservable(G_, GraphDomain.nodes)

class face_gds(gds):
	''' PDE defined on the faces of a graph ''' 
	pass		