import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import pdb

from gds.types import *
import gds

def harmonics(G, eps=1e-15):
	L = -nx.laplacian_matrix(G).todense()
	print('rank: ', np.linalg.matrix_rank(L))
	null = sp.linalg.null_space(L).T
	print(null.shape)

	sys = dict()
	for i, h in enumerate(null):
		obs = gds.node_gds(G)
		obs.set_evolution(nil=True)
		obs.set_initial(y0=lambda x: h[obs.X[x]])
		sys[f'harmonic_{i}'] = obs

	sys = gds.couple(sys)
	gds.render(sys, n_spring_iters=1000)

def spherical_harmonics():
	n = 20
	r = 1.0
	twopi = 2*np.pi
	delta = twopi / n
	nodes = []
	edges = []
	rnd = lambda x: x

	for i in range(n):
		for j in range(n):
			theta = delta * i
			phi = delta * j

			x = r*np.cos(theta)*np.cos(phi)
			y = r*np.sin(theta)*np.cos(phi)
			z = r*np.sin(phi)
			nodes.append(((rnd(x),rnd(y),rnd(z))))

			for i_ in [(i + 1) % n, (i - 1) % n]:
				theta_ = delta * i_
				x_ = r*np.cos(theta_)*np.cos(phi)
				y_ = r*np.sin(theta_)*np.cos(phi)
				z_ = r*np.sin(phi)
				edges.append(((rnd(x),rnd(y),rnd(z)), (rnd(x_),rnd(y_),rnd(z_))))

			for j_ in [(j + 1) % n, (j - 1) % n]:
				phi_ = delta * j_
				x_ = r*np.cos(theta)*np.cos(phi_)
				y_ = r*np.sin(theta)*np.cos(phi_)
				z_ = r*np.sin(phi_)
				edges.append(((rnd(x),rnd(y),rnd(z)), (rnd(x_),rnd(y_),rnd(z_))))

	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)

	harmonics(G)

def toroidal_harmonics():
	n = 20
	G = nx.grid_2d_graph(n, n, periodic=True)
	harmonics(G)

if __name__ == '__main__':
	spherical_harmonics()
	# toroidal_harmonics()