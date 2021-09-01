''' Wave equation ''' 

import numpy as np
import networkx as nx
import pdb
import colorcet as cc

import gds

def wave(G: nx.Graph, c: float=5.0):
	n = 20
	G = gds.triangular_lattice(n, n*2, periodic=True)
	amplitude = gds.node_gds(G)
	amplitude.set_evolution(dydt=lambda t, y: (c**2)*amplitude.laplacian(), order=2)
	amplitude.set_initial(y0=lambda x: x[0]*x[1]*(n-x[0])*(n-x[1]) / (n**3))
	gds.render(amplitude, dynamic_ranges=True, node_palette=cc.bmy, n_spring_iters=1000, title='Waves on a simplicial torus')

if __name__ == '__main__':
	# wave()
	