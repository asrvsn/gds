''' Swift-Hohenberg pattern formation ''' 

import numpy as np
import networkx as nx
import pdb
import random
import colorcet as cc

import gds

def swift_hohenberg(G: nx.graph, a: float, b: float, c: float, gam0: float, gam2: float) -> gds.node_gds:
	assert c > 0, 'Unstable'
	amplitude = gds.node_gds(G)
	amplitude.set_evolution(
		dydt=lambda t, y: -a*y - b*(y**2) - c*(y**3) + gam0*amplitude.laplacian(y) - gam2*amplitude.bilaplacian(y)
	)
	# amplitude.set_initial(y0=lambda _: np.random.uniform())
	amplitude.set_initial(y0=lambda x: x[0]+x[1])
	return amplitude

def stripes(G):
	return swift_hohenberg(G, 0.7, 0, 1, -2, 1)

def spots(G):
	return swift_hohenberg(G, 1-1e-2, -1, 1, -2, 1)

def spirals(G):
	return swift_hohenberg(G, 0.3, -1, 1, -2, 1)

if __name__ == '__main__':
	G = nx.hexagonal_lattice_graph(22, 23)
	eq = spirals(G)
	gds.render(eq, node_size=0.035, plot_width=800, node_palette=cc.bgy, dynamic_ranges=True, title='Pattern formation on a hexagonal lattice')