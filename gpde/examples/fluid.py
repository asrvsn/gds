import networkx as nx
import numpy as np
import pdb

from gpde.utils import set_seed
from gpde.core import *
from gpde.render.bokeh import *

def velocity_eq(G: nx.Graph, pressure: vertex_pde, kinematic_viscosity: float=1.0):
	def f(t, self):
		return -self.advect_self() - pressure.grad() + kinematic_viscosity * self.helmholtzian()
	return edge_pde(G, f)

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
	return pressure, velocity

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
	return pressure, velocity

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
	velocity.set_boundary(dirichlet=no_slip, dynamic=False)
	return pressure, velocity

def fluid_on_sphere():
	pass

def von_karman():
	w, h = 20, 10
	G = nx.grid_2d_graph(w, h)
	obstacle = [ # Introduce occlusion
		(6, 4), (6, 5), 
		(7, 4), (7, 5), 
		(8, 4),
	]
	G.remove_nodes_from(obstacle)
	def pressure_values(x):
		if x[0] == 0: return 1.0
		if x[0] == w-1: return -1.0
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
	return pressure, velocity

def random_graph():
	set_seed(1001)
	n = 30
	eps = 0.3
	G = nx.random_geometric_graph(n, eps)
	def pressure_values(x):
		if x == 5: return 1.0
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
	return pressure, velocity

if __name__ == '__main__':
	p, v = differential_inlets()
	sys = couple(p, v)
	# renderer = SingleRenderer(sys, node_rng=(-1,1))
	cycles = project_cycle_basis(v)
	renderer = CustomRenderer(sys[0], [[[[p, v]], [[c] for c in cycles]]], node_rng=(-1,1), colorbars=False)

	render_bokeh(renderer)
