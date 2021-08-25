''' Advection of scalar fields '''

import networkx as nx
import numpy as np
import colorcet as cc
import pdb 
import cProfile
import pstats

import gds
from gds.types import *

def advection(G, v_field, kind=None):
	flow_diff = np.zeros(len(G.edges()))
	flow = gds.edge_gds(G)
	flow.set_evolution(dydt=lambda t, y: flow_diff)
	flow.set_initial(y0 = v_field)
	conc = gds.node_gds(G)
	if kind == None:
		conc.set_evolution(dydt=lambda t, y: -conc.advect(flow))
	elif kind == 'lie':
		conc.set_evolution(dydt=lambda t, y: -conc.lie_advect(flow))
	else:
		raise Exception('unrecognized kind')
	return conc, flow

def advection_on_grid():
	n = 5
	G = gds.square_lattice(n, n)
	def v_field(e):
		if e[1][0] >= e[0][0] and e[1][1] >= e[0][1]:
			return 1
		else:
			return -1
	conc, flow = advection(G, v_field)
	conc.set_initial(y0 = lambda x: 1.0 if x == (0, 0) else 0.) # delta initial condition
	return conc, flow

def advection_on_triangles(periodic=False, v=1.0, **kwargs):
	m, n = 20, 20
	G = gds.triangular_lattice(m, n*2)
	if periodic:
		G.add_edges_from([((n, y), (0, y)) for y in range(m+1)])
		G.faces = [] # TODO: hacky; prevents infinite loop on face detection 
	def v_field(e):
		if e[0][1] == e[1][1]:
			if e[0][0] == 0 and e[1][0] == n:
				return -v
			if e[1][0] > e[0][0]:
				return v
		return 0.
	conc, flow = advection(G, v_field, **kwargs)
	conc.set_initial(y0 = lambda x: np.exp(-((x[0]-2)**2 + (x[1]-n/2)**2)/15)) 
	return conc, flow

def advection_on_random_graph():
	m, n = 20, 20
	G = nx.random_geometric_graph(100, 0.225)
	def v_field(e):
		return np.random.choice([-1., 1.])
	conc, flow = advection(G, v_field)
	conc.set_initial(y0 = lambda x: 1.) 
	return conc, flow

def advection_on_circle(v=1.0, **kwargs):
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	def v_field(e):
		if e == (n-1, 0) or e == (0, n-1):
			return -v
		return v
	conc, flow = advection(G, v_field, **kwargs)
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
	flow = gds.edge_gds(G)
	flow.set_evolution(dydt=lambda t, y: -flow.advect(vectorized=False))
	# flow.set_initial(y0=dict_fun({(2,3): 1.0, (3,4): 1.0}, def_val=0.))
	def init_flow(e):
		if e == (2,3): return 1.5
		elif e == (0,n-1): return -1.0
		return 1.0
	flow.set_initial(y0=init_flow)
	# flow.set_constraints(dirichlet=dict_fun({(2,3): 1.0}))
	return flow

def vector_advection_circle_2():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	flow = gds.edge_gds(G)
	obs = gds.edge_gds(G)
	flow.set_evolution(dydt=lambda t, y: np.zeros_like(y))
	obs.set_evolution(dydt=lambda t, y: -obs.advect(flow, vectorized=False))
	def init_obs(e):
		if e == (2,3): return 1.5
		elif e == (0,n-1): return -1.0
		return 1.0
	obs.set_initial(y0=init_obs)
	def init_flow(e):
		if e == (0,n-1): return -1.0
		return 1.0
	flow.set_initial(y0=init_flow)
	# flow.set_constraints(dirichlet=dict_fun({(2,3): 1.0}))
	return flow, obs

def vector_advection_test(flows=[1,1,1,1,1], **kwargs):
	G = nx.Graph()
	G.add_nodes_from([0, 1, 2, 3, 4, 5])
	G.add_edges_from([(2, 0), (3, 0), (0, 1), (1, 5), (1, 4)])
	flow = gds.edge_gds(G)
	def field(e):
		ret = 1
		if e == (0, 2):
			ret = -1
		if e == (0, 3):
			ret = -1
		return flows[flow.edges[e]] * ret
	flow.set_evolution(dydt=lambda t, y: np.zeros_like(y))
	flow.set_initial(y0=field)
	obs = gds.edge_gds(G)
	obs.set_evolution(dydt=lambda t, y: -obs.advect(**kwargs))
	obs.set_initial(y0=field)
	return flow, obs

def vector_advection_test_suite():
	# Test 1a
	for _ in range(1000):
		v_field = [np.random.choice([-1, 1]) for _ in range(5)]
		v, u = vector_advection_test(v_field)
		u.advect(v_field=v, check=True)

	# Test 1b
	for _ in range(1000):
		v_field = [np.random.uniform(-1, 1) for _ in range(5)]
		v, u = vector_advection_test(v_field)
		u.advect(v_field=v, check=True)

	# Test 2
	v_field = [1,1,1,1,1]
	v, u = vector_advection_test(v_field, check=True)
	u.step(1.0)

def self_advection_1():
	G = nx.Graph()
	G.add_nodes_from(list(range(1,7)))
	G.add_edges_from([
		(1,2),(2,3),(3,4),(4,1),
		(4,5),(5,6),(6,3),
	])
	negated = set([(1,4),(3,6),])
	def v_field(e):
		ret = 1.0
		if e in negated:
			ret *= -1
		if e == (3,4):
			ret *= 2
		return ret
	u = gds.edge_gds(G)
	u.set_evolution(dydt=lambda t, y: -u.advect())
	u.set_initial(y0=v_field)
	return u

def self_advection_2():
	G = nx.Graph()
	G.add_nodes_from(list(range(1,9)))
	v_field = {
		(1,2): 2, (2,3): 1, (3,4): 2, (1,4): -3,
		(5,6): 2, (6,7): 3, (7,8): 2, (5,8): -1,
		(1,5): 1, (2,6): 1, (3,7): -1, (4,8): -1,
	}
	G.add_edges_from(v_field.keys())
	u = gds.edge_gds(G)
	u.set_evolution(dydt=lambda t, y: -u.advect())
	u.set_initial(y0=lambda e: v_field[e])
	return u

def scalar_advection_kinds_test():
	conc1, flow1 = advection_on_triangles(periodic=True)
	# conc1, flow1 = advection_on_circle()
	conc2, flow2 = advection_on_triangles(periodic=True, kind='lie')
	# conc2, flow2 = advection_on_circle()
	sys = gds.couple({
		'conc1': conc1,
		'flow1': flow1,
		'div1': flow1.project(GraphDomain.nodes, lambda v: v.div()),
		'conc2': conc2,
		'flow2': flow2,
		'div2': flow2.project(GraphDomain.nodes, lambda v: v.div()),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), dynamic_ranges=True, node_size=.05, title='Advection of a Gaussian concentration')

if __name__ == '__main__':
	''' Scalar field advection ''' 

	# conc, flow = advection_on_triangles()
	# sys = gds.couple({
	# 	'conc': conc,
	# 	'flow': flow,
	# })
	# gds.render(sys, canvas=[[[[conc, flow]]]], dynamic_ranges=True, node_size=.05, title='Advection of a Gaussian concentration')

	# scalar_advection_kinds_test()

	# conc, flow = advection_on_circle(v=1.0)
	# sys = gds.couple({
	# 	'conc': conc,
	# 	'flow': flow,
	# })
	# gds.render(sys, canvas=[[[[conc, flow]]]], dynamic_ranges=True, plot_height=600, node_size=.05, min_rng_size=0.05, y_rng=(-1.1,0.8), title='Advection on a circle')


	# conc, flow = advection_on_random_graph()
	# sys = gds.couple({
	# 	'conc': conc,
	# 	'flow': flow,
	# })
	# gds.render(sys, canvas=[[[[conc, flow]]]], node_rng=(0.5, 1.5), colorbars=False, plot_height=600, node_size=.05, y_rng=(-1.1,1.1), title='Absorbing points of an initially uniform mass')

	''' Vector field advection ''' 

	vector_advection_test_suite()

	# cProfile.run('vector_advection_test_suite()', 'out.prof')
	# prof = pstats.Stats('out.prof')
	# prof.sort_stats('time').print_stats(40)

	# v_1, u_1 = vector_advection_test([1,1,1,1,1], vectorized=False)
	# v_2, u_2 = vector_advection_test([1,1,1,1,-1], vectorized=False)
	# v_3, u_3 = vector_advection_test([-1,1,1,1,-1], vectorized=False)
	# sys = gds.couple({
	# 	'u_1': u_1,
	# 	'v_1': v_1,
	# 	'u_2': u_2,
	# 	'v_2': v_2,
	# 	'u_3': u_3,
	# 	'v_3': v_3,
	# })
	# canvas = [
	# 	[[[u_1]], [[u_2]], [[u_3]]],
	# 	[[[v_1]], [[v_2]], [[v_3]]],
	# ]
	# gds.render(sys, canvas=canvas, dynamic_ranges=True, edge_max=1.0, title='Advection of a vector field')

	# flow, obs = vector_advection_circle_2()
	# sys = gds.couple({'flow': flow, 'obs': obs})
	# gds.render(sys, edge_rng=(0,1.5), dynamic_ranges=True)

	# flow = vector_advection_circle()
	# gds.render(flow, edge_max=0.5, edge_rng=(0,1.5), min_rng_size=0.05)

	# flow = self_advection_2()
	# sys = gds.couple({
	# 	'flow': flow,
	# 	'divergence': flow.project(gds.GraphDomain.nodes, lambda v: v.div()),
	# 	'L1': flow.project(PointObservable, lambda v: np.abs(v.y).sum()),
	# 	'KE': flow.project(PointObservable, lambda v: 0.5*np.dot(v.y, v.y)),
	# })
	# gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 2), edge_max=0.5, edge_rng=(0,2), dynamic_ranges=True, min_rng_size=0.05)

