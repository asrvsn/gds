import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict
import itertools

from .types import *
from .fds import *
from .utils import *
from .utils.graph import embedded_faces, get_edge_weights, get_planar_mesh, degree_matrix

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

		if hasattr(G, 'faces'): # TODO: hacky
			faces = G.faces
		else:
			faces, self.outer_face = embedded_faces(G)
		self.faces = {f: i for i, f in enumerate(faces)}
		self.faces_i = {i: f for f, i in self.faces.items()}

		# Weights
		w_map = get_edge_weights(G)
		self.edge_weights = np.array([w_map[e] for e in self.edges])

		# Orientations
		self.edge_orientation = {**{e: 1 for e in self.edges}, **{(e[1], e[0]): -1 for e in self.edges}} # Orientation implicit by stored keys in domain
		self.face_orientation = {f: 1 for f in self.faces} # Orientation of faces; 1: CCW, -1: CW
		self.face_orientation_vector = np.array([self.face_orientation[f] for f in self.faces])

		if Gd is GraphDomain.nodes:
			X = self.nodes
		elif Gd is GraphDomain.edges:
			X = self.edges
		elif Gd is GraphDomain.triangles:
			X = self.triangles
		elif Gd is GraphDomain.faces:
			X = self.faces
		Observable.__init__(self, X)

	def project(self, other: Union[GraphDomain, Observable], view: Callable[['GraphObservable'], np.ndarray], *args, **kwargs) -> 'GraphObservable':
		if type(other) is GraphDomain:
			return super().project(GraphObservable, view, self.G, other)
		elif type(other) is nx.Graph:
			canary = GraphObservable(other, self.Gd)
			indices = []
			for i in range(canary.ndim):
				x = canary.iX[i]
				indices.append(self.X[x])
			indices = np.array(indices, dtype=np.intp)
			# TODO: ignores view argument
			view = lambda obs: self.y[indices]
			return super().project(GraphObservable, view, other, self.Gd)
		else:
			return super().project(other, view, *args, **kwargs)

	# TODO: hacky
	def t(self):
		return

	def y(self):
		return


	''' Meshing '''

	def mesh(self) -> dict:
		mesh = get_planar_mesh(self.G)

		if self.Gd is GraphDomain.nodes:
			mesh = np.array([mesh[k] for k in self.nodes])
			return mesh
		elif self.Gd is GraphDomain.edges:
			mesh = np.array([
				((mesh[k[0]][0] + mesh[k[1]][0])/2, (mesh[k[0]][1] + mesh[k[1]][1])/2) for k in self.edges
			])
			return mesh
		elif self.Gd is GraphDomain.faces:
			mesh = np.array([
				(np.mean([mesh[n][0] for n in k]), np.mean([mesh[n][1] for n in k])) for k in self.faces
			])
			return mesh
		else: 
			raise Exception('unsupported domain for mesh()')

''' Dynamical systems on generic graph domains ''' 

class gds(fds, GraphObservable):
	def __init__(self, G: nx.Graph, Gd: GraphDomain):
		GraphObservable.__init__(self, G, Gd)
		fds.__init__(self, self.X)

		self.incidence = (nx.incidence_matrix(G, oriented=True)@sp.diags(np.sqrt(self.edge_weights))).tocsr() # |V| x |E| incidence


	def set_constraints(self, *args, **kwargs):
		# TODO: better way to handle constraints on >=1-dimensional objects (need to detect alternating signs)
		fds.set_constraints(self, *args, **kwargs)

		if self.Gd is GraphDomain.nodes:
			self.dirichlet_laplacian = self.vertex_laplacian.copy()
			self.dirichlet_laplacian[self.dirichlet_indices, :] = 0
			self.dirichlet_laplacian.eliminate_zeros()
			self.neumann_correction[self.neumann_indices] = self.neumann_values
		# elif self.Gd is GraphDomain.edges:
		# 	self.dirichlet_laplacian = self.edge_laplacian.copy()
		# 	self.dirichlet_laplacian[self.dirichlet_indices, :] = 0
		# 	self.dirichlet_laplacian.eliminate_zeros()
			# TODO: neumann conditions

		if self.iter_mode is IterationMode.cvx:
			# Rebuild cost function since operators may have changed
			self.rebuild_cvx()

''' Dynamical systems on specific graph domains ''' 

class node_gds(gds):
	''' Dynamical system defined on the nodes of a graph ''' 

	def __init__(self, G: nx.Graph, *args, **kwargs):
		gds.__init__(self, G, GraphDomain.nodes, *args, **kwargs)

		# Operators
		self.vertex_laplacian = -self.incidence@self.incidence.T # |V| x |V| laplacian operator
		self.dirichlet_laplacian = self.vertex_laplacian
		self.neumann_correction = np.zeros(self.ndim)	

	''' Differential operators: all of the following are CVXPY-compatible '''

	def partial(self, e: Edge) -> float:
		return np.sqrt(self.edge_weights[self.edges[e]]) * (self(e[1]) - self(e[0])) 

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
		'''
		Transportation of a scalar field by a general (non-divergence-free) vector field.
		'''
		if isinstance(v_field, edge_gds):
			assert v_field.G is self.G, 'Incompatible domains'
			v_field = v_field.y
		if y is None: y=self.y
		Bp = self.incidence@sp.diags(np.sign(v_field))
		Bp.data[Bp.data > 0] = 0.
		Bp.data *= -1
		return -self.incidence@sp.diags(v_field)@Bp.T@y

	def lie_advect(self, v_field: Union[Callable[[Edge], float], np.ndarray], y: np.ndarray=None) -> np.ndarray:
		'''
		Lie advection by a divergence-free vector field 
		'''
		if isinstance(v_field, edge_gds):
			assert v_field.G is self.G, 'Incompatible domains'
			v_field = v_field.y
		if y is None: y=self.y
		Bp = self.incidence@sp.diags(np.sign(v_field))
		Bp.data[Bp.data < 0] = 0.
		return Bp@sp.diags(v_field)@self.incidence.T@y


class edge_gds(gds):
	''' Dynamical system defined on the edges of a graph ''' 
	def __init__(self, G: nx.Graph, v_free=None, *args, **kwargs):
		gds.__init__(self, G, GraphDomain.edges, *args, **kwargs)

		self.edge_laplacian = -self.incidence.T@self.incidence # |E| x |E| laplacian operator
		self.dirichlet_laplacian = self.edge_laplacian
		self.neumann_correction = np.zeros(self.ndim)	

		''' Additional operators '''

		# Edge-edge adjacency matrix
		def edge_adj_func(e_i, e_j):
			if e_i == e_j:
				return 0
			elif e_i[0] == e_j[0] or e_i[1] == e_j[1]:
				return -1
			elif e_i[1] == e_j[0] or e_i[0] == e_j[1]:
				return 1
			return 0
		self.edge_adj = sparse_product(self.edges.keys(), self.edges.keys(), edge_adj_func).tocsr() # |E| x |E| edge adjacency matrix

		def simplicial_curl(tri, edge):
			# TODO: proper weighting
			if edge[0] in tri and edge[1] in tri:
				c = np.sqrt(self.edge_weights[self.edges[edge]])
				if edge == (tri[0], tri[1]) or edge == (tri[1], tri[2]) or edge == (tri[2], tri[0]): # Orientation of triangle
					return c
				return -c
			return 0
		if len(self.triangles) > 0:
			self.curl3 = sparse_product(self.triangles.keys(), self.edges.keys(), simplicial_curl).tocsr() # |T| x |E| curl operator, where T is the set of 3-cliques in G; respects implicit orientation
		else:
			self.curl3 = sp.csr_matrix((1, len(self.edges)))

		def geometric_curl(face, edge):
			# TODO: proper weighting
			if edge[0] in face and edge[1] in face:
				c = np.sqrt(self.edge_weights[self.edges[edge]])
				if any([edge == (face[i-1], face[i]) for i in range(1, len(face))]) or edge == (face[-1], face[0]):
					return c
				return -c
			return 0
		if len(self.faces) > 0:
			self.curl_face = sparse_product(self.faces.keys(), self.edges.keys(), geometric_curl).tocsr() # |F| x |E| curl operator, where F is the set of faces in G; respects implicit orientation
			# self.curl_outer_face = sparse_product([self.outer_face], self.edges.keys(), geometric_curl).tocsr() # 1 x |E| curl operator, where F is the set of faces in G; respects implicit orientation
		else:
			self.curl_face = sp.csr_matrix((1, len(self.edges)))

		self.G_ = nx.line_graph(G)

		if v_free is None:
			B = self.incidence
		else:
			B = self.incidence.copy()
			for v in v_free:
				B.data[B.indptr[v]:B.indptr[v+1]] = 0
		self.leray_projector = np.asarray(np.eye(self.ndim) - self.incidence.T@np.linalg.pinv((self.incidence@self.incidence.T).todense(), hermitian=True)@B)
		# self.leray_projector = (self.leray_projector + self.leray_projector.T) / 2 # Symmetrize

		# Dual weights
		Dinv = degree_matrix(G)
		data = Dinv.data
		data = 1/(data-1)
		data[data == np.inf] = 0
		Dinv.data = data
		B = abs(self.incidence)
		self.dual_weights = B.T@Dinv@B
		self.dual_weights.setdiag(0)
		self.dual_weights.eliminate_zeros()
		self.dual_weights = self.dual_weights.tocsr()

	def __call__(self, x: Edge):
		return self.edge_orientation[x] * self.y[self.X[x]]

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
		return self.curl_face@y

	def dd_(self, y: np.ndarray=None, normal: np.ndarray=None) -> np.ndarray:
		if y is None: y=self.y
		ret = -self.incidence.T@self.incidence@y
		ret[self.dirichlet_indices] = 0
		if not (normal is None): ret[normal] = 0
		return ret

	def d_d(self, y: np.ndarray=None) -> np.ndarray:
		if y is None: y=self.y
		ret = -self.curl_face.T@self.curl_face@y
		ret[self.dirichlet_indices] = 0
		return ret

	def laplacian(self, y: np.ndarray=None, free: np.ndarray=None, normal: np.ndarray=None) -> np.ndarray:
		''' 
		Hodge 1-Laplacian
		TODO: neumann conditions
		''' 
		ret = self.dd_(y=y, normal=normal) + self.d_d(y=y)
		if not (free is None):
			ret[free] = 0
		return ret

	def bilaplacian(self, y: np.ndarray=None) -> np.ndarray:
		''' 
		TODO: neumann conditions
		TODO: check curl term?
		''' 
		return self.laplacian(self.laplacian(y))

	def advect(self, y: np.ndarray=None, stiff=200, weighted=True) -> np.ndarray:
		'''
		Nearest-neighbors advective derivative of an edge flow.
		'''
		if y is None: y=self.y

		S = sp.diags(np.sign(y))
		Y = sp.diags(y)
		Y_ = sp.diags(np.abs(y))

		B = self.incidence@S
		Bm = -B.copy()
		Bp = B.copy()
		Bm.data[Bm.data < 0] = 0
		Bp.data[Bp.data < 0] = 0
		P = Bm.T@Bm - Bp.T@Bp
		M = (Y_@P - P@Y_)
		M.data = np.tanh(stiff * M.data)
		F = Bm.T@Bp - Bp.T@Bm # + M
		if weighted: 
			F = F.multiply(self.dual_weights)
		A = Y@F@S + S@F@Y

		return -A@y

	def advection_tensor(self, y: np.ndarray=None, stiff=200, weighted=True) -> np.ndarray:
		'''
		Advective interaction 3-tensor 
		'''
		if y is None: y=self.y

		S = sp.diags(np.sign(y))
		Y = sp.diags(y)
		Y_ = sp.diags(np.abs(y))

		B = self.incidence@S
		Bm = -B.copy()
		Bp = B.copy()
		Bm.data[Bm.data < 0] = 0
		Bp.data[Bp.data < 0] = 0
		P = Bm.T@Bm - Bp.T@Bp
		M = (Y_@P - P@Y_)
		M.data = np.tanh(stiff * M.data)
		F = Bm.T@Bp - Bp.T@Bm + M
		if weighted: 
			F = F.multiply(self.dual_weights)
		F = S@F@S
		T = []
		for i in range(y.size):
			selector = np.zeros(y.size)
			selector[i] = 1
			selector = sp.diags(selector)
			T.append((selector@F + F@selector).todense())
		T = np.array(T)
		return T


	def lie_advect(self, v_field: Union[Callable[[Edge], float], np.ndarray], y: np.ndarray=None) -> np.ndarray:
		'''
		Lie advection by a divergence-free vector field 
		'''
		if y is None: y=self.y
		if v_field is None: 
			v_field = self.y
		elif isinstance(v_field, edge_gds):
			assert v_field.G is self.G, 'Incompatible domains'
			# Since graphs are identical, orientation is implicitly respected
			v_field = v_field.y
			
		U2 = self.face_curl@sp.diags(v_field)
		Bp.data[Bp.data < 0] = 0.
		return Bp@self.incidence.T@y

	def leray_project(self, y: np.ndarray=None) -> np.ndarray:
		"""
		Project onto divergence-free subspace subject to boundary conditions.
		"""
		if y is None: y=self.y
		if self.dirichlet_indices.size == 0:
			return self.leray_projector@y
		else:
			A = self.incidence@self.incidence.T
			b = self.incidence@y
			A = sp.vstack([A, self.incidence.T[self.dirichlet_indices, :]])
			b = np.concatenate((b, y[self.dirichlet_indices]))
			x = sp.linalg.lsmr(A, b)[0]
			return y - self.incidence.T@x

	def vertex_dual(self) -> GraphObservable:
		''' View the vertex-edge dual graph ''' 
		edge_map = np.array([self.X[e] for e in self.G_.nodes], dtype=np.intp)
		class DualGraphObservable(GraphObservable):
			@property
			def y(other):
				return self.y[edge_map]
			@property
			def t(other):
				return self.t
		return DualGraphObservable(self.G_, GraphDomain.nodes)

	def edge_neighbors(self, e):
		if e in self.G_: return self.G_.neighbors(e)
		else: return self.G_.neighbors((e[1], e[0]))


class simplex_gds(gds):
	''' Dynamical system defined on k-simplices of a graph ''' 
	pass

class face_gds(gds):
	''' Dynamical system defined on the faces of a graph ''' 

	def __init__(self, G: nx.Graph, *args, **kwargs):
		gds.__init__(self, G, GraphDomain.faces, *args, **kwargs)

		# TODO: dry
		def geometric_curl(face, edge):
			# TODO: proper weighting
			if edge[0] in face and edge[1] in face:
				c = np.sqrt(self.edge_weights[self.edges[edge]])
				if any([edge == (face[i-1], face[i]) for i in range(1, len(face))]) or edge == (face[-1], face[0]):
					return c
				return -c
			return 0
		if len(self.faces) > 0:
			self.curl_face = sparse_product(self.faces.keys(), self.edges.keys(), geometric_curl).tocsr() # |F| x |E| curl operator, where F is the set of faces in G; respects implicit orientation
			# self.curl_outer_face = sparse_product([self.outer_face], self.edges.keys(), geometric_curl).tocsr() # 1 x |E| curl operator, where F is the set of faces in G; respects implicit orientation
		else:
			self.curl_face = sp.csr_matrix((1, len(self.edges)))

		# Operators: TODO

	def laplacian(self, y: np.ndarray=None) -> np.ndarray:
		''' 
		Hodge 2-Laplacian
		TODO: boundary conditions
		''' 
		if y is None: y=self.y
		return -self.curl_face@self.curl_face.T@y
