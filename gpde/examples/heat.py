''' Heat equation ''' 

import numpy as np
import networkx as nx
import pdb

from gpde import *
from gpde.render.bokeh import *

def heat_grid(n = 10) -> vertex_pde:
	G = nx.grid_2d_graph(n, n)
	f = lambda t, self: self.laplacian()
	return vertex_pde(G, f)

def grid_const_boundary() -> vertex_pde:
	n = 10
	temperature = heat_grid(n=n)
	temperature.set_boundary(dirichlet = lambda t, x: 1.0 if (0 in x or (n-1) in x) else None)
	return temperature

def grid_mixed_boundary() -> vertex_pde:
	n = 10
	temperature = heat_grid(n=n)
	def dirichlet(t, x):
		if x[0] == 0 or x[0] == n-1:
			return 0.5
		elif x[1] == 0:
			return 1.0
		return None
	def neumann(t, x):
		if x[1] == n-1 and x[0] not in (0, n-1):
			return -0.1
		return None
	temperature.set_boundary(dirichlet=dirichlet, neumann=neumann)
	return temperature

def grid_timevarying_boundary() -> vertex_pde:
	n = 10
	temperature = heat_grid(n=n)
	temperature.set_boundary(
		dirichlet = lambda t, x: np.sin(t/5)**2 if (0 in x or (n-1) in x) else None
	)
	return temperature

if __name__ == '__main__':
	# Use coupling to visualize multiple PDEs simultaneously
	system = couple(grid_const_boundary(), grid_mixed_boundary())
	render_bokeh(GridRenderer(system))
