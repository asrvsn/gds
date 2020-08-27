import networkx as nx
import numpy as np

from gpde.core import *
from gpde.render.bokeh import *

def fluid_on_grid():
	n = 10
	G = nx.grid_2d_graph(n, n)
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0 = lambda t, x: 1. if x == (2,2) else None)
	velocity = edge_pde(G, lambda t, self: -self.advect(self) - pressure.grad())
	return coupled_pde(pressure, velocity)

def fluid_on_sphere():
	pass

def von_karman():
	pass

if __name__ == '__main__':
	sys = fluid_on_grid().system()
	render_bokeh(GridRenderer(sys))
