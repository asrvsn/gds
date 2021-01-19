''' Boundary condition utilities ''' 

import networkx as nx
from typing import Callable, List

def const_velocity(dG: nx.Graph, v: float) -> Callable:
	''' Create constant-velocity condition graph boundary ''' 
	def vel(e):
		if e in dG.edges or (e[1], e[0]) in dG.edges: 
			return v
		return None
	return vel

def no_slip(dG: nx.Graph) -> Callable:
	''' Create no-slip velocity condition graph boundary ''' 
	return const_velocity(dG, 0.)

def combine_bc(bcs: List[Callable]) -> Callable:
	def fun(x):
		for bc in bcs:
			v = bc(x)
			if v is not None: return v
		return None
	return fun