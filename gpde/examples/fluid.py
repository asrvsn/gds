import networkx as nx
import numpy as np

from gpde.core import *
from gpde.render.bokeh import *

def fluid_on_grid():
	n = 8
	G = nx.grid_2d_graph(n, n)
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	def pressure_values(x):
		if x == (3,3):
			return 1.0
		if x == (5,5):
			return -1.0
		return 0.
	pressure.set_initial(y0=pressure_values)
	velocity = edge_pde(G, lambda t, self: -self.advect(self) - pressure.grad())
	return couple(pressure, velocity)

def fluid_on_sphere():
	pass

def von_karman():
	pass

if __name__ == '__main__':
	sys = fluid_on_grid()
	render_bokeh(SingleRenderer(sys))
