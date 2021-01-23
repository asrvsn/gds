''' Boundary condition utilities ''' 

import networkx as nx
from typing import Callable, List, Tuple

from .common import *
from gds.types import *

def const_edge_bc(dG: nx.Graph, v: float) -> BoundaryCondition:
	''' Create constant-velocity condition along edges of graph boundary ''' 
	def vel(e):
		if e in dG.edges or (e[1], e[0]) in dG.edges: 
			return v
		return None
	return vel

def zero_edge_bc(dG: nx.Graph) -> BoundaryCondition:
	''' Create no-slip velocity condition graph boundary ''' 
	return const_edge_bc(dG, 0.)

def combine_bcs(*bcs: Tuple[BoundaryCondition]) -> BoundaryCondition:
	bcs = [dict_fun(bc) if type(bc) is dict else bc for bc in bcs]
	x_bcs = [bc for bc in bcs if fun_ary(bc) == 1]
	tx_bcs = [bc for bc in bcs if fun_ary(bc) == 2]
	if len(tx_bcs) > 0:
		def fun(t, x):
			for bc in x_bcs:
				v = bc(x)
				if v is not None: return v
			for bc in tx_bcs:
				v = bc(t, x)
				if v is not None: return v
			return None
		return fun
	else:
		def fun(x):
			for bc in x_bcs:
				v = bc(x)
				if v is not None: return v
			return None
		return fun