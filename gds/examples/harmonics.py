import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import pdb
import colorcet as cc

from gds.types import *
import gds

def show_harmonics(G, k=0, n=np.inf, **kwargs):
	def make_observable():
		# TODO: use the simplicial decomposition here
		return {
			0: gds.node_gds,
			1: gds.edge_gds,
			2: gds.face_gds,
		}[k](G)

	obs = make_observable()
	L = obs.laplacian(np.eye(obs.ndim))
	print('Laplacian rank: ', np.linalg.matrix_rank(L))
	null = sp.linalg.null_space(L).T

	sys = dict()
	for i, h in enumerate(null):
		if i < n:
			obs = make_observable()
			obs.set_evolution(nil=True)
			obs.set_initial(y0=lambda x: h[obs.X[x]])
			sys[f'harmonic_{i}'] = obs

	sys = gds.couple(sys)
	gds.render(sys, n_spring_iters=1000, title=f'{k}-harmonics', **kwargs)

if __name__ == '__main__':
	# G = gds.icosphere()
	# G = gds.icotorus(n=10)
	G = gds.torus()

	# nx.draw(G, nx.spring_layout(G, iterations=500))
	# plt.show()

	# show_harmonics(G, k=0)
	show_harmonics(G, k=1, dynamic_ranges=True, edge_colors=True, edge_palette=cc.bmy)
	# show_harmonics(G, k=2, dynamic_ranges=True, face_palette=cc.bmy)