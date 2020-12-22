import numpy as np 
from typing import Tuple, Dict, Any
from scipy.integrate import RK45
import pdb
from itertools import count

from .core import *
from .utils import *


''' Other derived observables for measurement ''' 

class MetricsObservable(Observable):
	''' Observable for measuring scalar derived quantities ''' 
	def __init__(self, base: Observable, metrics: Dict):
		self.base = base
		self.metrics = metrics
		X = dict(zip(metrics.keys(), count()))
		super().__init__(self, X)

	@property 
	def t(self):
		return self.base.t

	@property 
	def y(self):
		''' Call in order to update metrics. TODO: brittle? ''' 
		self.metrics = self.calculate(self.base.y, self.metrics)
		return self.metrics

	@abstractmethod
	def calculate(self, y: Any, metrics: Dict):
		''' Override to calculate metrics set ''' 
		pass


''' Domain-specific gpde's ''' 

class vertex_pde(gpde):
	''' PDE defined on the vertices of a graph ''' 

	def __init__(self, G: nx.Graph, *args, **kwargs):
		gpde.__init__(self, G, GraphDomain.vertices, *args, **kwargs)

	''' Differential operators '''

	def partial(self, e: Edge) -> float:
		return np.sqrt(self.weights[self.edges[e]]) * (self(e[1]) - self(e[0])) 

	def grad(self) -> np.ndarray:
		return self.incidence.T@self.y

	def laplacian(self) -> np.ndarray:
		''' Dirichlet-Neumann Laplacian. TODO: should minimize error from laplacian on interior? ''' 
		ret = self.vertex_laplacian@self.y
		ret[self.neumann_indices] += self.neumann_values
		ret[self.dirichlet_indices] = 0.
		return ret

	def bilaplacian(self) -> np.ndarray:
		# TODO correct way to handle Neumann in this case? (Gradient constraint only specifies one neighbor beyond)
		return self.vertex_laplacian@self.laplacian()

	def advect(self, v_field: Callable[[Edge], float]) -> np.ndarray:
		# TODO: optimize for shared graph
		return np.array([
			sum([v_field((x, y)) * self.partial((x, y)) for y in self.G.neighbors(x)])
			for x in self.X
		])

class edge_pde(gpde):
	''' PDE defined on the edges of a graph ''' 
	def __init__(self, G: nx.Graph, *args, **kwargs):
		gpde.__init__(self, G, GraphDomain.edges, *args, **kwargs)
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

	def curl(self) -> np.ndarray:
		raise self.curl3@self.y

	def laplacian(self) -> np.ndarray:
		''' Edge laplacian 
		https://www.stat.uchicago.edu/~lekheng/work/psapm.pdf 
		TODO: neumann conditions
		''' 
		# return -self.curl3.T@self.curl3@self.y
		# ret = self.edge_laplacian@self.y
		# ret[self.dirichlet_indices] = 0.
		# return ret
		return self.edge_laplacian@self.y

	def helmholtzian(self) -> np.ndarray:
		''' Vector laplacian or discrete Helmholtz operator 
		https://www.stat.uchicago.edu/~lekheng/work/psapm.pdf 
		TODO: neumann conditions
		''' 
		return self.laplacian() - self.curl3.T@self.curl3@self.y

	def advect(self, v_field: Callable[[Edge], float] = None) -> np.ndarray:
		if v_field is None:
			return (self.adj_dual@self.y)*self.y
		elif type(v_field) is edge_pde and v_field.G is self.G:
			# Since graphs are identical, orientation is implicitly respected
			return self.y * (self.adj_dual@v_field.y)
		else:
			# TODO: check correctness
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

class face_pde(gpde):
	''' PDE defined on the faces of a graph ''' 
	pass		

''' Other derivations ''' 

class coupled_pde(Integrable):
	''' Multiple PDEs coupled in time. 
	Can be integrated together but not observed directly. 
	Can couple both forward-solved and direct-solved PDEs.
	''' 
	def __init__(self, *pdes: Tuple[pde], max_step=None):
		assert len(pdes) >= 2, 'Pass two or more pdes to couple'
		assert all([p.t == 0. for p in pdes]), 'Pass pdes at initial values only'
		self.t0 = 0.
		self.forward_pdes = []
		self.direct_pdes = []
		for p in pdes:
			if p.mode is SolveMode.forward:
				self.forward_pdes.append(p)
			else:
				self.direct_pdes.append(p)
		self.has_forward = len(self.forward_pdes) > 0

		if self.has_forward:
			if max_step is None:
				max_step = min([p.max_step for p in self.forward_pdes])
			self.max_step = max_step
			y0s = [p.y0 for p in self.forward_pdes]
			self.views = [slice(0, len(y0s[0]))]
			for i in range(1, len(self.forward_pdes)):
				start = self.views[i-1].stop
				self.views.append(slice(start, start+len(y0s[i])))
			self.y0 = np.concatenate(y0s)
			self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=max_step)
			# Patch all PDEs to refer to values from current integrator (TODO: better way...?)
			for (p, view) in zip(self.forward_pdes, self.views):
				p.view = view
				attach_dyn_props(p, {'y': lambda p: self.integrator.y[p.view], 't': lambda _: self.integrator.t})
			for p in self.direct_pdes:
				attach_dyn_props(p, {'t': lambda _: self.integrator.t})


	def dydt(self, t: Time, y: np.ndarray):
		for p in self.direct_pdes:
			p.step_direct(0.) # Interleave direct solvers with forward solvers
		res = np.concatenate([p.dydt(t, y[view]) for (p, view) in zip(self.forward_pdes, self.views)])
		return res

	def step(self, dt: float):
		if self.has_forward:
			self.integrator.t_bound = self.t + dt
			self.integrator.status = 'running'
			while self.integrator.status != 'finished':
				self.integrator.step()
				# Apply all boundary conditions
				for p, view in zip(self.forward_pdes, self.views):
					if p.dynamic_bc:
						for x in p.dirichlet_X:
							self.integrator.y[view][p.X[x] - p.ndim] = p.dirichlet(self.integrator.t, x)
		else:
			# In the case of no forward-solved PDE's, this class is merely a utility for simultaneously solving direct PDE's
			for p in self.direct_pdes:
				p.step_direct(dt)

	def observables(self) -> List[Observable]:
		return self.forward_pdes + self.direct_pdes

	def system(self) -> System:
		return (self, self.observables())

	def reset(self):
		if self.has_forward:
			self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=self.max_step)
		for p in self.direct_pdes:
			p.reset()

	@property
	def t(self):
		if self.has_forward:
			return self.integrator.t
		else:
			return self.direct_pdes[0].t