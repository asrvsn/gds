from typing import Callable
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import pdb
import colorcet as cc

from gds.types import *
import gds

def show_leray(G, v: Callable=None, **kwargs):
	'''
	Show Leray decomposition of a vector field.
	'''
	if v is None:
		v = lambda x: np.random.uniform(1, 2)
	orig = gds.edge_gds(G)
	orig.set_evolution(nil=True)
	orig.set_initial(y0=v)

	div_free = orig.project(GraphDomain.edges, lambda u: u.leray_project())
	curl_free = orig.project(GraphDomain.edges, lambda u: u.y - u.leray_project())

	sys = gds.couple({
		'original': orig,
		'original (div)': orig.project(GraphDomain.nodes, lambda u: u.div()),
		'original (curl)': orig.project(GraphDomain.faces, lambda u: u.curl()),
		'div-free': div_free,
		'div-free (div)': div_free.project(GraphDomain.nodes, lambda u: orig.div(u.y)), # TODO: chaining projections?
		'div-free (curl)': div_free.project(GraphDomain.faces, lambda u: orig.curl(u.y)),
		'curl-free': curl_free,
		'curl-free (div)': curl_free.project(GraphDomain.nodes, lambda u: orig.div(u.y)),
		'curl-free (curl)': curl_free.project(GraphDomain.faces, lambda u: orig.curl(u.y)),
	})

	gds.render(sys, n_spring_iters=1000, canvas=gds.grid_canvas(sys.observables.values(), 3), title='Leray decomposition', min_rng_size=1e-3, **kwargs)

if __name__ == '__main__':
	# G = gds.square_lattice(10, 10)
	# G = gds.icosphere()
	# G = gds.icotorus(n=12)
	G = gds.torus()
	# G = gds.k_torus(2)

	# nx.draw(G, nx.spring_layout(G, iterations=1000))
	# plt.show()

	show_leray(G, dynamic_ranges=True, edge_colors=True, edge_palette=cc.bmy, edge_max=0.5)
