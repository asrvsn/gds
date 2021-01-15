''' Advection of scalar fields '''

import networkx as nx
import pdb

from gds import *
from gds.render.bokeh import *
from gds.utils.graph import *

def advection(G, v_field) -> (node_gds, edge_gds):
	flow_diff = np.zeros(len(G.edges()))
	flow = edge_gds(G, dydt=lambda t, self: flow_diff)
	flow.set_initial(y0 = v_field)
	conc = node_gds(G, dydt=lambda t, self: -self.advect(flow))
	return conc, flow

def advection_on_grid():
	n = 5
	G = grid_graph(n, n)
	def v_field(e: Edge):
		if e[1][0] >= e[0][0] and e[1][1] >= e[0][1]:
			return 1
		else:
			return -1
	conc, flow = advection(G, v_field)
	conc.set_initial(y0 = lambda x: 1.0 if x == (0, 0) else 0.) # delta initial condition
	return conc, flow

def advection_on_circle():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	def v_field(e: Edge):
		if e == (n-1, 0) or e == (0, n-1):
			return -1.0
		return 1.0
	conc, flow = advection(G, v_field)
	conc.set_initial(y0 = lambda x: 1.0 if x == 0 else 0.) # delta initial condition
	return conc, flow

def advection_on_torus():
	n = 20
	G = grid_graph(n, n, periodic=True)
	def v_field(e: Edge):
		if e[0][1] == e[1][1]:
			if e[0][0] > e[1][0] or e[1][0] == (e[0][0] - n - 1):
				return 1
			else:
				return -1
		else:
			return 0
	flow_diff = np.zeros(len(G.edges()))
	flow = edge_gds(G, lambda t, self: flow_diff)
	flow.set_initial(y0 = v_field)
	conc = node_gds(G, f = lambda t, self: self.advect(v_field))
	conc.set_initial(y0 = lambda x: 1.0 if x == (10, 10) else None) # delta initial condition
	return couple(conc, flow)

def test():
	G = nx.Graph()
	G.add_nodes_from([1, 2, 3, 4, 5])
	edges = [(1, 2), (3, 2), (4, 3), (5, 4), (1, 5), (1, 3)]
	G.add_edges_from(edges)
	v_field = lambda e: 1.0 if e in edges else -1.0
	conc, flow = advection(G, v_field)
	pdb.set_trace()
	return conc, flow

def vector_advection_circle():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	flow = edge_gds(G, dydt=lambda t, self: -self.advect())
	# flow.set_initial(y0=dict_fun({(2,3): 1.0, (3,4): 1.0}, def_val=0.))
	flow.set_initial(y0=lambda e: 1.0 if e == (2, 3) else 0.1)
	# flow.set_constraints(dirichlet=dict_fun({(2,3): 1.0}))
	return flow

if __name__ == '__main__':
	''' Scalar field advection ''' 

	# conc, flow = advection_on_grid()
	# sys = System(couple(conc, flow), {
	# 	'conc': conc,
	# 	'flow': flow,
	# })
	# renderer = LiveRenderer(sys, [[[[conc, flow]]]], node_palette=cc.rainbow, node_rng=(-1,1), node_size=0.03)
	# renderer.start()

	''' Vector field advection ''' 

	flow = vector_advection_circle()
	sys = System(flow, {
		'flow': flow,
	})
	renderer = LiveRenderer(sys, [[[[flow]]]], node_palette=cc.rainbow, node_rng=(-1,1), node_size=0.03)
	renderer.start()
