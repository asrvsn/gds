''' Specifying partial differential equations on graph domains ''' 

import numpy as np
import networkx as nx
import networkx.linalg.graphmatrix as nx
from typing import Any, Union, Tuple, Callable, Newtype, Iterable
from scipy.integrate import RK45
from abc import ABC, abstractmethod

from .utils import *

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

	def __call__(self, x: Point):
		''' Measure at a point '''
		return self.y[self.X[x]]

''' Common types ''' 

Vertex = Any
Edge = Tuple[Vertex, Vertex]
Face = Tuple[Vertex, ...]
Point = Union[Vertex, Edge, Face] # A point in the graph domain
Time = Newtype('Time', float)
Domain = Dict[Point, int] # Mapping from points into array indices
Sign = NewType('Sign', int)
System = Tuple[Integrable, List[Observable]]

''' Base classes ''' 

class pde(Observable, Integrable):
	def __init__(self, X: Domain, f: Callable[[Time], np.ndarray], order: int=1, max_step=None):
		''' Create a PDE defined on some domain.
		Uses RK45 solver.
		''' 
		Observable.__init__(self, X)
		self.Xi = list(X.values())
		self.ndim = len(X)
		self.t0 = 0.
		self.y0 = np.zeros(self.ndim*order)
		self.f = f
		self.order = order
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=max_step)
		self.dirichlet = lambda t, x: None
		self.dirichlet_X = set()
		self.dirichlet_indices = []
		self.neumann = lambda t, x: None
		self.neumann_X = set()
		self.neumann_indices = []
		self.erroneous = lambda y: False 

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
			if self.erroneous(self.y):
				raise ValueError('Erroneous state encountered')

	def reset(self):
		''' Reset the system to initial conditions ''' 
		pass

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

class multi_pde(Integrable):
	''' Multiple PDEs coupled in time. Can be integrated together but not observed directly. ''' 
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
		# Patch all PDEs to refer to values from current integrator (TODO: better way...?)
		from (p, view) in zip(self.pdes, self.views):
			def y(other):
				return self.integrator.y[view]
			# TODO
			p.y = property(y)
		self.y0 = np.concatenate(y0s)
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=max_step)

	def dydt(self, t: Time, y: np.ndarray):
		return np.concatenate([p.dydt(t, y[view]) for (p, view) in zip(self.pdes, self.views)])

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

	def system(self) -> System:
		return (self, self.observables())

	def reset(self):
		self.integrator = RK45(self.dydt, self.t0, self.y0, np.inf, max_step=self.max_step)

