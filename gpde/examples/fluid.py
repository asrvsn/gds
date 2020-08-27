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

def differential_inlets():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4,5])
	G.add_edges_from([(1,2),(2,3),(4,3),(2,5),(3,5)])
	def pressure_values(x):
		if x == 1: return 1.0
		if x == 4: return 0.5
		if x == 5: return -0.5
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = edge_pde(G, lambda t, self: -self.advect(self) - pressure.grad())
	return couple(pressure, velocity)

def poiseuille():
	G = nx.grid_2d_graph(10, 5)
	def pressure_values(x):
		if x[0] == 0: return 1.0
		if x[0] == 9: return -1.0
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = edge_pde(G, lambda t, self: -self.advect(self) - pressure.grad())
	def no_slip(t, x):
		if x[1] == 0 or x[1] == 4:
			return 0.
		return None
	velocity.set_boundary(dirichlet=no_slip)
	return couple(pressure, velocity)

def fluid_on_sphere():
	pass

def von_karman():
	pass

if __name__ == '__main__':
	sys = poiseuille()
	render_bokeh(SingleRenderer(sys))
