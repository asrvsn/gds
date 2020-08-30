import networkx as nx
import numpy as np

from gpde.core import *
from gpde.render.bokeh import *

def velocity_eq(G: nx.Graph, pressure: vertex_pde, dyn_viscosity: float=1.0) -> edge_pde:
	return edge_pde(G, lambda t, self: -self.advect(self) - pressure.grad() + dyn_viscosity * self.laplacian())

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
	velocity = velocity_eq(G, pressure)
	return couple(pressure, velocity)

def differential_inlets():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4,5,6])
	G.add_edges_from([(1,2),(2,3),(4,3),(2,5),(3,5),(5,6)])
	def pressure_values(x):
		if x == 1: return 1.0
		if x == 4: return 0.5
		if x == 6: return -0.5
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
	# velocity.set_initial(y0=lambda x: 1e-2)
	return couple(pressure, velocity)

def poiseuille():
	G = nx.grid_2d_graph(10, 5)
	def pressure_values(x):
		if x[0] == 0: return 1.0
		if x[0] == 9: return -1.0
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
	def no_slip(t, x):
		if x[0][1] == x[1][1] == 0 or x[0][1] == x[1][1] == 4:
			return 0.
		return None
	velocity.set_boundary(dirichlet=no_slip)
	return couple(pressure, velocity)

def fluid_on_sphere():
	pass

def von_karman():
	pass

if __name__ == '__main__':
	sys = fluid_on_grid()
	render_bokeh(SingleRenderer(sys))
