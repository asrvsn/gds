''' Operations on pde's ''' 

from typing import List, Tuple
import networkx as nx

from .core import *
from .derived import *

def couple(*pdes: Tuple[pde]) -> System:
	return coupled_pde(*pdes).system()

# TODO: fix
# def project_cycle_basis(p: gpde) -> List[EdgeObservable]:
# 	class ProjObservable(EdgeObservable):
# 		def __init__(self, cycle: list):
# 			self.G = nx.Graph()
# 			nx.add_cycle(self.G, cycle)
# 			self.edge_indices = [p.edges[e] for e in self.G.edges()]
# 			self.orientation = np.array([p.orientation[e] for e in self.G.edges()])
# 		@property
# 		def t(self):
# 			return p.t
# 		@property
# 		def y(self):
# 			return p.y[self.edge_indices] * self.orientation
		
# 	return [ProjObservable(cycle) for cycle in nx.cycle_basis(p.G)]
