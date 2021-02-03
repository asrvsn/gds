import numpy as np
import networkx as nx
from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict

from .types import *
from .fds import *
from .utils import *

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
		if Gd is GraphDomain.nodes:
			self.dirichlet_laplacian = self.vertex_laplacian
		elif Gd is GraphDomain.edges:
			self.dirichlet_laplacian = self.edge_laplacian
		self.neumann_correction = np.zeros(self.ndim)	
		def curl_element(tri, edge):
			if edge[0] in tri and edge[1] in tri:
				c = np.sqrt(self.weights[self.edges[edge]])
				if edge == (tri[0], tri[1]) or edge == (tri[1], tri[2]) or edge == (tri[2], tri[0]): # Orientation of triangle
					return c
				return -c
			return 0
		self.curl3 = sparse_product(self.triangles.keys(), self.edges.keys(), curl_element) # |T| x |E| curl operator, where T is the set of 3-cliques in G; respects implicit orientation

		fds.__init__(self, self.X)

	def set_constraints(self, *args, **kwargs):
		fds.set_constraints(self, *args, **kwargs)

		if self.Gd is GraphDomain.nodes:
			self.dirichlet_laplacian = self.vertex_laplacian.copy()
			self.dirichlet_laplacian[self.dirichlet_indices, :] = 0
			self.dirichlet_laplacian.eliminate_zeros()
			self.neumann_correction[self.neumann_indices] = self.neumann_values
		else:
			self.dirichlet_laplacian = self.edge_laplacian.copy()
			self.dirichlet_laplacian[self.dirichlet_indices, :] = 0
			self.dirichlet_laplacian.eliminate_zeros()
			# TODO: neumann conditions

		if self.iter_mode is IterationMode.cvx:
			# Rebuild cost function since operators may have changed
			self.rebuild_cvx()

''' Dynamical systems on specific graph domains ''' 

class node_gds(gds):
	''' Dynamical system defined on the nodes of a graph ''' 

	def __init__(self, G: nx.Graph, *args, **kwargs):
		gds.__init__(self, G, GraphDomain.nodes, *args, **kwargs)

	''' Differential operators: all of the following are CVXPY-compatible '''

	def partial(self, e: Edge) -> float:
		return np.sqrt(self.weights[self.edges[e]]) * (self(e[1]) - self(e[0])) 

	def grad(self, y: np.ndarray=None) -> np.ndarray:
		if y is None: y=self.y
		return self.incidence.T@y

	def laplacian(self, y: np.ndarray=None) -> np.ndarray:
		''' Dirichlet-Neumann Laplacian. TODO: should minimize error from laplacian on interior? ''' 
		if y is None: y=self.y
		return self.dirichlet_laplacian@y + self.neumann_correction

	def bilaplacian(self, y: np.ndarray=None) -> np.ndarray:
		# TODO correct way to handle Neumann in this case? (Gradient constraint only specifies one neighbor beyond)
		if y is None: y=self.y
		return self.dirichlet_laplacian@self.laplacian(y)

	def advect(self, v_field: Union[Callable[[Edge], float], np.ndarray], y: np.ndarray=None) -> np.ndarray:
		# TODO: check application of dirichlet conditions
		if isinstance(v_field, edge_gds):
			assert v_field.G is self.G, 'Incompatible domains'
			v_field = v_field.y
		if y is None: y=self.y
		M = self.incidence.multiply(v_field)
		N = M.copy().T
		N.data[N.data > 0] = 0.
		N.data *= -1
		ret = -M@N@y
		ret[self.dirichlet_indices] = 0.
		return ret

class edge_gds(gds):
	''' Dynamical system defined on the edges of a graph ''' 
	def __init__(self, G: nx.Graph, *args, **kwargs):
		gds.__init__(self, G, GraphDomain.edges, *args, **kwargs)

	def __call__(self, x: Edge):
		return self.orientation[x] * self.y[self.X[x]]

	''' Differential operators: all of the following are CVXPY-compatible '''

	def div(self, y: np.ndarray=None) -> np.ndarray:
		if y is None: y=self.y
		return -self.incidence@y

	def influx(self, y: np.ndarray=None) -> np.ndarray:
		''' In-flux through nodes ''' 
		if y is None: y=self.y
		f = self.incidence.multiply(y)
		f.data[f.data < 0] = 0.
		return f.sum(axis=1)

	def outflux(self, y: np.ndarray=None) -> np.ndarray:
		''' Out-flux through nodes ''' 
		if y is None: y=self.y
		f = -self.incidence.multiply(y)
		f.data[f.data < 0] = 0.
		return f.sum(axis=1)

	def curl(self, y: np.ndarray=None) -> np.ndarray:
		if y is None: y=self.y
		raise self.curl3@y

	def laplacian(self, y: np.ndarray=None) -> np.ndarray:
		''' Vector laplacian or discrete Helmholtz operator or Hodge-1 laplacian
		https://www.stat.uchicago.edu/~lekheng/work/psapm.pdf 
		TODO: neumann conditions
		TODO: check curl term?
		''' 
		if y is None: y=self.y
		return self.dirichlet_laplacian@y - self.curl3.T@self.curl3@y

	def bilaplacian(self, y: np.ndarray=None) -> np.ndarray:
		''' 
		TODO: neumann conditions
		TODO: check curl term?
		''' 
		return self.laplacian(self.laplacian(y))

	def advect(self, v_field: Union[Callable[[Edge], float], np.ndarray] = None, y: np.ndarray=None) -> np.ndarray:
		'''
		Adjoint of scalar advection case.
		# TODO: does this assume flow is irrotational?
		# TODO: check application of dirichlet conditions
		'''
		if y is None: y=self.y
		if v_field is None:
			M = self.incidence.multiply(y).T
			N = M.copy().T
			M.data[M.data > 0] = 0.
			M.data *= -1
			ret = -M@N@y
			ret[self.dirichlet_indices] = 0.
			return ret
		else:
			if isinstance(v_field, edge_gds):
				assert v_field.G is self.G, 'Incompatible domains'
				# Since graphs are identical, orientation is implicitly respected
				v_field = v_field.y
			M = self.incidence.multiply(v_field).T
			N = M.copy().T
			M.data[M.data > 0] = 0.
			M.data *= -1
			ret = -M@N@y
			ret[self.dirichlet_indices] = 0.
			return ret

	def vertex_dual(self) -> GraphObservable:
		''' View the vertex-dedge dual graph ''' 
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
	''' Dynamical system defined on the faces of a graph ''' 
	pass		