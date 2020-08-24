import numpy as np 
from typing import Tuple
from scipy.integrate import RK45

from .core import *

''' Common types ''' 

System = Tuple[Integrable, List[Observable]]

''' Operations / views on gpde's ''' 

def couple(*pdes: Tuple[gpde]) -> System:
	sys = multi_gpde(pdes)
	return sys, sys.observables()
