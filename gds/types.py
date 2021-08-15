from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict, List
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

''' Graph domains ''' 

Node = Any
Edge = Tuple[Node, Node]
Triangle = Tuple[Node, Node, Node]
Face = List[Node]
Point = Union[Node, Edge, Triangle, Face] # A point in the graph domain

''' Types for dynamics on graph domains '''

Time = NewType('Time', float)
Domain = Dict[Point, int] # Mapping from points into array indices
Sign = NewType('Sign', int)


class GraphDomain(Enum):
	nodes = 0
	edges = 1
	triangles = 2
	faces = 3

Orientation = Dict[Edge, Sign]

BoundaryCondition = Union[
	Dict[Point, float],
	Callable[[Point], float],
	Callable[[Time, Point], float]
]

''' Base interfaces ''' 

class IterationMode(Enum):
	none = 0 # Unspecified
	dydt = 1 # Differential
	cvx = 2 # Convex program
	map = 3 # Recurrence relation
	traj = 4 # Recorded trajectory
	nil = 5 # Nil iteration

class Steppable(ABC):
	''' An object which can be stepped through time ''' 
	def __init__(self, iter_mode: IterationMode):
		self.iter_mode = iter_mode

	@abstractmethod
	def step(self, dt: float):
		''' Step the system to t+dt ''' 
		pass

	@abstractmethod
	def reset(self):
		pass

class Observable(ABC):
	''' An object which can be observed through time ''' 
	def __init__(self, X: Domain):
		self.X = X # Domain
		self.iX = {i: x for x, i in X.items()} # Reverse-lookup domain
		self.ndim = len(X)

	@property
	@abstractmethod
	def t(self) -> float:
		pass

	@property
	@abstractmethod
	def y(self) -> np.ndarray:
		# TODO check dimensions
		pass

	def __getitem__(self, idx):
		return self.y.__getitem__(idx)

	def __call__(self, x: Point):
		''' Measure at a point '''
		return self.y[self.X[x]]

	def __len__(self):
		return self.ndim

	def project(self, other: type, view: Callable[['Observable'], np.ndarray], *args, **kwargs):
		class Projection(other):
			def __init__(projected_self):
				other.__init__(projected_self, *args, **kwargs)
			@property
			def t(projected_self):
				return self.t
			@property
			def y(projected_self):
				return view(self)
		return Projection()

class PointObservable(Observable):
	'''
	Observe values on a zero-dimensional space
	'''
	def __init__(self, retention=600, min_res=1e-3, **kwargs):
		self.render_params = {
			'retention': retention,
			'min_res': min_res,
			**kwargs
		} # TODO seoarate rendering
		super().__init__(dict())

