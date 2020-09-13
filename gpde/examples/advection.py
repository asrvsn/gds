''' Advection of scalar fields '''

import networkx as nx
import pdb

from gpde import *
from gpde.render.bokeh import *

def advection_on_grid():
	n = 5
	G = nx.grid_2d_graph(n, n)
	def v_field(e: Edge):
		if e[0][1] == e[1][1]:
			if e[0][0] > e[1][0]:
				return 1
			else:
				return -1
		return 0
	flow_diff = np.zeros(len(G.edges()))
	flow = edge_pde(G, lambda t, self: flow_diff)
	flow.set_initial(y0 = v_field)
	concentration = vertex_pde(G, f = lambda t, self: -self.advect(v_field))
	concentration.set_initial(y0 = lambda x: 1.0 if x == (2, 2) else 0.) # delta initial condition
	return couple(concentration, flow)

def advection_on_circle():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	def v_field(e: Edge):
		if e[1] > e[0] or e == (n-1, 0):
			return 1.0
		return -1.0
	flow_diff = np.zeros(len(G.edges()))
	flow = edge_pde(G, lambda t, self: flow_diff)
	flow.set_initial(y0 = v_field)
	concentration = vertex_pde(G, f = lambda t, self: -self.advect(v_field))
	concentration.set_initial(y0 = lambda x: 1.0 if x == 0 else 0.) # delta initial condition
	return couple(concentration, flow)

def advection_on_torus():
	n = 20
	G = nx.grid_2d_graph(n, n, periodic=True)
	def v_field(e: Edge):
		if e[0][1] == e[1][1]:
			if e[0][0] > e[1][0] or e[1][0] == (e[0][0] - n - 1):
				return 1
			else:
				return -1
		else:
			return 0
	flow_diff = np.zeros(len(G.edges()))
	flow = edge_pde(G, lambda t, self: flow_diff)
	flow.set_initial(y0 = v_field)
	concentration = vertex_pde(G, f = lambda t, self: self.advect(v_field))
	concentration.set_initial(y0 = lambda x: 1.0 if x == (10, 10) else None) # delta initial condition
	return couple(concentration, flow)

if __name__ == '__main__':
	sys = advection_on_circle()
	SingleRenderer(sys).start()
