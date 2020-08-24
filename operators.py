import numpy as np 
from typing import Tuple
from scipy.integrate import RK45

from .core import *

''' Operations / views on gpde's ''' 

def couple(*xs: Tuple[RK45]):
	''' Couple multiple integrators in time ''' 
	assert all([x.t == xs[0].t for x in xs]), 'Cannot couple integrators at different times'
	pass