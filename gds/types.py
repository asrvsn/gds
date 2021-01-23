from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

''' Graph domains ''' 

Node = Any
Edge = Tuple[Node, Node]
Triangle = Tuple[Node, Node, Node]
Point = Union[Node, Edge, Triangle] # A point in the graph domain

''' Types for dynamics on graph domains '''

Time = NewType('Time', float)
Domain = Dict[Point, int] # Mapping from points into array indices
Sign = NewType('Sign', int)


class GraphDomain(Enum):
	nodes = 0
	edges = 1
	triangles = 2

Orientation = Dict[Edge, Sign]

BoundaryCondition = Union[
	Dict[Point, float],
	Callable[[Point], float],
	Callable[[Time, Point], float]
]

''' Base interfaces ''' 

class IterationMode(Enum):
	none = 0
	dydt = 1 # Differential
	cvx = 2 # Convex program
	map = 3 # Recurrence relation
	traj = 4 # Recorded trajectory

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
		pass

	def __getitem__(self, idx):
		return self.y.__getitem__(idx)

	def __call__(self, x: Point):
		''' Measure at a point '''
		return self.y[self.X[x]]

	def __len__(self):
		return self.ndim