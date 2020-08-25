''' Heat equation ''' 

import numpy as np
import networkx as nx

from gpde.core import *
from gpde.ops import *
from gpde.render.bokeh import *

def heat_grid(n = 10) -> vertex_pde
	n = 8
	G = nx.grid_2d_graph(n, n)
	temperature = vertex_pde(G, lambda t, self: self.laplacian())
	return temperature

def grid_const_boundary() -> vertex_pde:
	temperature = heat_grid()
	temperature.set_boundary(dirichlet = lambda t, x: 1.0 if (0 in x or (n-1) in x) else None)
	return temperature

def grid_mixed_boundary() -> vertex_pde:
	temperature = heat_grid()
	temperature.set_boundary(
		dirichlet = lambda t, x: 1.0 if (0 in x) else None,
		neumann = lambda t, x: -0.1 if ((n-1) in x) else None
	)
	return temperature

if __name__ == '__main__':
	# Use coupling to visualize multiple PDEs simultaneously
	system = couple(grid_const_boundary(), grid_mixed_boundary())
	render_bokeh(GridRenderer(system))