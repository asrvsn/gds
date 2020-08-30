import numpy as np 
from typing import Tuple
from scipy.integrate import RK45
import pdb

from .core import *
from .utils import *

''' Derived interfaces ''' 

class VertexObservable(Observable):
	def __init__(self, G: nx.Graph):
		self.G = G
		X = {x: i for i, x in enumerate(G.nodes())}
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
		# pdb.set_trace()
		VertexObservable.__init__(self, G)
		self.X_edge = bidict({x: i for i, x in enumerate(G.edges())})
		self.X_from = [self.X[e[0]] for e in G.edges()]
		self.X_to = [self.X[e[1]] for e in G.edges()]
		self.laplacian_matrix = -nx.laplacian_matrix(G)
		self.weights = np.ones(len(G.edges()))
		if w_key is not None:
			for i, e in enumerate(G.edges()):
				self.weights[i] = G[e[0]][e[1]][w_key]
		pde.__init__(self, self.X, *args, **kwargs)

	''' Spatial differential operators '''

	def partial(self, e: Edge) -> float:
		i = self.X_edge[e]
		return np.sqrt(self.weights[i]) * (self(e[1]) - self(e[0])) 

	def grad(self) -> np.ndarray:
		# Respects implicit orientation
		return np.sqrt(self.weights) * (self.y[self.X_to] - self.y[self.X_from]) 

	def laplacian(self) -> np.ndarray:
		fixed_flux = replace(np.zeros(self.y.shape), self.neumann_indices, [self.neumann(self.t, x) for x in self.neumann_X])
		return self.laplacian_matrix@self.y + fixed_flux

	def bilaplacian(self) -> np.ndarray:
		# TODO correct way to handle Neumann in this case? (Gradient constraint only specifies one neighbor beyond)
		return self.laplacian_matrix@self.laplacian()

	def advect(self, v_field: Callable[[Edge], float]) -> np.ndarray:
		return np.array([
			sum([v_field((x, y)) * self.partial((x, y)) for y in self.G.neighbors(x)])
			for x in self.X
		])


class edge_pde(pde, EdgeObservable):
	''' PDE defined on the edges of a graph ''' 
	def __init__(self, G: nx.Graph, *args, w_key: str=None, **kwargs):
		EdgeObservable.__init__(self, G)
		self.X_vertex = {x: i for i, x in enumerate(G.nodes())}
		self.weights = np.ones(len(G.edges()))
		if w_key is not None:
			for i, e in enumerate(G.edges()):
				self.weights[i] = G[e[0]][e[1]][w_key]
		self.incidence = nx.incidence_matrix(G)
		# Orient edges
		self.orientation = {**{x: 1 for x in self.X}, **{(x[1], x[0]): -1 for x in self.X}} # Orientation implicit by stored keys in domain
		# self.oriented_incidence = self.incidence.copy()
		# for i, x in enumerate(G.edges()):
		# 	(a, b) = x
		# 	if self.orientation[i] > 0:
		# 		self.oriented_incidence[self.X_vertex[a], i] = -1
		# 	else:
		# 		self.oriented_incidence[self.X_vertex[b], i] = -1
		# Vertex dual
		self.G_dual = nx.line_graph(G)
		self.X_dual = bidict({x: i for i, x in enumerate(self.G_dual.edges())})
		self.dual_laplacian_matrix = -nx.laplacian_matrix(self.G_dual)
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
		pde.__init__(self, self.X, *args, **kwargs)

	def __call__(self, x: Edge):
		return self.orientation[x] * self.y[self.X[x]]

	''' Spatial differential operators '''

	def div(self) -> np.ndarray:
		return 2 * np.sqrt(self.weights) * self.y@self.incidence.T

	def curl(self) -> np.ndarray:
		raise NotImplementedError

	def laplacian(self) -> np.ndarray:
		''' Vector laplacian https://en.wikipedia.org/wiki/Vector_Laplacian ''' 
		# TODO: check? (also need edge-edge weights...)
		# TODO: neumann conditions
		return self.dual_laplacian_matrix@self.y

	def advect(self, v_field: Callable[[Edge], float]) -> np.ndarray:
		ret = np.zeros(self.ndim)
		for a, i in self.X.items():
			u = self(a)
			for b in self.G_dual.neighbors(a):
				w = self.weights_dual[self.X_dual[(a, b)]]
				if b[0] == a[1]: # Outgoing edge
					ret[i] += u * v_field(b) / w
				elif b[1] == a[1]: # Outgoing edge, reversed direction
					ret[i] += u * v_field((b[1], b[0])) / w
				elif b[0] == a[0]: # Ingoing edge, reversed directopm
					ret[i] -= u * v_field((b[1], b[0])) / w
				else: # Ingoing edge
					ret[i] -= u * v_field(b) / w
		return np.array(ret)

class face_pde(pde, FaceObservable):
	''' PDE defined on the faces of a graph ''' 
	pass		

''' Other derivations ''' 

class coupled_pde(Integrable):
	''' Multiple PDEs coupled in time. Can be integrated together but not observed directly. ''' 
	def __init__(self, *pdes: Tuple[pde], max_step=None):
		assert len(pdes) >= 1 
		assert all([p.t == 0. for p in pdes]), 'Pass pdes at initial values only'
		self.pdes = pdes
		if max_step is None:
			max_step = min([p.max_step for p in pdes])
		self.max_step = max_step
		self.t0 = 0.
		y0s = [p.y0 for p in pdes]
		self.views = [slice(0, len(y0s[0]))]
		for i in range(1, len(pdes)):
			start = self.views[i-1].stop
			self.views.append(slice(start, start+len(y0s[i])))
		self.y0 = np.concatenate(y0s)
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=max_step)
		# Patch all PDEs to refer to values from current integrator (TODO: better way...?)
		for (p, view) in zip(self.pdes, self.views):
			p.view = view
			attach_dyn_props(p, {'y': lambda p: self.integrator.y[p.view], 't': lambda _: self.integrator.t})

	def dydt(self, t: Time, y: np.ndarray):
		return np.concatenate([p.dydt(t, y[view]) for (p, view) in zip(self.pdes, self.views)])

	def step(self, dt: float):
		self.integrator.t_bound = self.t + dt
		self.integrator.status = 'running'
		while self.integrator.status != 'finished':
			self.integrator.step()
			# Apply all boundary conditions
			for p, view in zip(self.pdes, self.views):
				for x in p.dirichlet_X:
					self.integrator.y[view][p.X[x] - p.ndim] = p.dirichlet(self.integrator.t, x)

	def observables(self) -> List[Observable]:
		return list(self.pdes)

	def system(self) -> System:
		return (self, self.observables())

	def reset(self):
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=self.max_step)

	@property
	def t(self):
		return self.integrator.t