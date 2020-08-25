''' Operations on pde's ''' 

from .core import *

def couple(*pdes: Tuple[pde]) -> System:
	sys = multi_gpde(pdes)
	return sys, sys.observables()

def project_cycle_basis(p: pde) -> System:
	pass
