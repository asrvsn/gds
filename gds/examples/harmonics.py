import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import pdb
import colorcet as cc
import random

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
	L = -obs.laplacian(np.eye(obs.ndim))
	print('Laplacian rank: ', np.linalg.matrix_rank(L))
	null = sp.linalg.null_space(L).T

	sys = dict()
	sys['Surface'] = gds.face_gds(G)
	sys['Surface'].set_evolution(nil=True)
	for i, h in enumerate(null):
		if i < n:
			obs = make_observable()
			obs.set_evolution(nil=True)
			obs.set_initial(y0=lambda x: h[obs.X[x]])
			sys[f'harmonic_{i}'] = obs

	sys = gds.couple(sys)
	gds.render(sys, n_spring_iters=1000, title=f'L{k}-harmonics', **kwargs)

def show_L0_eigfuns(G, n=12, **kwargs):
	'''
	Show eigenfunctions of 0-Laplacian
	'''
	obs = gds.node_gds(G)
	# L = np.array(nx.laplacian_matrix(G).todense())
	L = -obs.laplacian(np.eye(obs.ndim))
	# vals, vecs = sp.sparse.linalg.eigs(-L, k=n, which='SM')
	vals, vecs = np.linalg.eig(L)
	vals, vecs = np.real(vals), np.real(vecs)
	# pdb.set_trace()
	sys = dict()
	sys['Surface'] = gds.face_gds(G)
	sys['Surface'].set_evolution(nil=True)
	canvas = dict()
	canvas['Surface'] = [[[sys['Surface']]]]
	for i, (ev, vec) in enumerate(sorted(zip(vals, vecs.T), key=lambda x: np.abs(x[0]))):
		ev = np.round(ev, 5)
		obs = gds.node_gds(G)
		obs.set_evolution(nil=True)
		obs.set_initial(y0=lambda x: vec[obs.X[x]])
		if ev in canvas:
			sys[f'eigval_{ev} eigfun_{len(canvas[ev])}'] = obs
			canvas[ev].append([[obs]])
		else:
			sys[f'eigval_{ev} eigfun_{0}'] = obs
			canvas[ev] = [[[obs]]]
		if i == n:
			break
	# canvas = sorted(list(canvas.values()), key=len)
	canvas = list(canvas.values())
	sys = gds.couple(sys)
	gds.render(sys, canvas=canvas, n_spring_iters=1200, title=f'L0-eigenfunctions', **kwargs)


def square_defect(G, v):
	i, j = v
	gds.remove_face(G, v)
	f = ((i-1,j),(i-1,j-1),(i,j-1),(i+1,j-1),(i+1,j),(i+1,j+1),(i,j+1),(i-1,j+1))
	G.faces.append(f)

def defective_sphere():
	G = gds.icosphere()
	gds.remove_face(G, 137)
	f = (120,140,37,139,53,0)
	G.faces.append(f)
	return G

if __name__ == '__main__':
	gds.set_seed(1)

	# Harmonics on genus-k surfaces
	# G = gds.icosphere()
	# G = gds.torus()
	# G = gds.icotorus()
	# G = gds.k_torus(2, m=8, n=11)
	# G = gds.k_torus(3, m=8, n=11)

	# Harmonics with topological defect
	# G = defective_sphere()
	# G = gds.torus()
	# square_defect(G, (8,8))
	# square_defect(G, (2,5))

	# Harmonics with degenerate edge
	G = gds.k_torus(2, m=8, n=11, degenerate=True)

	# u = gds.face_gds(G)
	# u.set_evolution(nil=True)
	# gds.render(u)

	# nx.draw(G, nx.spring_layout(G, iterations=1000))
	# plt.show()

	# show_harmonics(G, k=0)
	show_harmonics(G, k=1, edge_colors=True, edge_palette=cc.bmy)
	# show_harmonics(G, k=2, face_palette=cc.bmy)

	# show_L0_eigfuns(G, n=16, node_palette=cc.bmy)