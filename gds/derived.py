import numpy as np 
from typing import Tuple, Dict, Any
from scipy.integrate import RK45, LSODA
import scipy.sparse as sp
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


