''' Operations on pde's ''' 

from typing import List, Tuple
import networkx as nx
import shortuuid
import os.path
import os
import hickle as hkl
import cloudpickle
from tqdm import tqdm

from .core import *
from .derived import *

def couple(*pdes: Tuple[pde]) -> System:
	return coupled_pde(*pdes)

def solve_and_dump(sys: System, T: float, dt: float=1e-3, folder='dump', parent='runs'): 
	assert os.path.isdir(parent), f'Parent directory "{parent}" does not exist'
	path = parent + '/' + folder
	if not os.path.isdir(path):
		os.mkdir(path)
	dump = dict()
	for name, obs in sys.observables.items():
		dump[name] = []
	t = 0.
	with tqdm(total=int(T / dt)) as pbar:
		while t < T:
			sys.integrator.step(dt)
			for name, obs in sys.observables.items():
				dump[name].append(obs.y.copy())
			t += dt
			pbar.update(1)
	# Dump simulation data
	for name, data in dump.items():
		hkl.dump(np.array(data), f'{path}/{name}.hkl', mode='w', compression='gzip')
	# Dump system object
	with open(f'{path}/system.pkl', 'wb') as f:
		sys.dt = dt # Tell the dt (hacky)
		cloudpickle.dump(sys, f)


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
