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
	'''
	Show harmonics of Hodge k-Laplacian
	'''
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

def show_L0_eigfuns(G, n=12, **kwargs):
	'''
	Show eigenfunctions of 0-Laplacian
	'''
	obs = gds.node_gds(G)
	L = obs.laplacian(np.eye(obs.ndim))
	vals, vecs = sp.sparse.linalg.eigs(-L, k=n, which='SM')
	sys = dict()
	canvas = dict()
	for i in range(n):
		ev = np.round(vals[i], 4)
		obs = gds.node_gds(G)
		obs.set_evolution(nil=True)
		obs.set_initial(y0=lambda x: vecs[obs.X[x], i])
		if ev in canvas:
			sys[f'eigval_{ev} eigfun_{len(canvas[ev])}'] = obs
			canvas[ev].append([[obs]])
		else:
			sys[f'eigval_{ev} eigfun_{0}'] = obs
			canvas[ev] = [[[obs]]]
	canvas = sorted(list(canvas.values()), key=len)
	sys = gds.couple(sys)
	gds.render(sys, canvas=canvas, n_spring_iters=1000, title=f'L0-eigenfunctions', **kwargs)


if __name__ == '__main__':
	# G = gds.icosphere()
	# G = gds.icotorus(n=12)
	G = gds.torus()
	# G = gds.k_torus(2)

	# nx.draw(G, nx.spring_layout(G, iterations=1000))
	# plt.show()

	# show_harmonics(G, k=0)
	# show_harmonics(G, k=1, dynamic_ranges=True, edge_colors=True, edge_palette=cc.bmy)
	# show_harmonics(G, k=2, dynamic_ranges=True, face_palette=cc.bmy)

	show_L0_eigfuns(G, n=8, dynamic_ranges=True)