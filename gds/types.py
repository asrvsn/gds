from typing import Tuple, Any, Union
from enum import Enum
import numpy as np

''' Types for dynamics on generic domains '''

Time = NewType('Time', float)
Domain = Dict[Point, int] # Mapping from points into array indices
Sign = NewType('Sign', int)

''' Graph domains ''' 

Vertex = Any
Edge = Tuple[Vertex, Vertex]
Triangle = Tuple[Vertex, Vertex, Vertex]
Point = Union[Vertex, Edge, Triangle] # A point in the graph domain

class GraphDomain(Enum):
	nodes = 0
	edges = 1
	triangles = 2

Orientation = Dict[Edge, Sign]

''' Base interfaces ''' 

class IterationMode(Enum):
	none = 0
	dydt = 1
	cvx = 2
	map = 3

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