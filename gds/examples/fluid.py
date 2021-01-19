import networkx as nx
import numpy as np
import pdb
from itertools import count
import colorcet as cc
import random
import cvxpy as cp

from gds import *
from gds.utils import set_seed
from gds.utils.graph import *
from gds.utils.boundary import *
from gds.render.bokeh import *

''' Definitions ''' 

def incompressible_flow(G: nx.Graph, dG: nx.Graph, viscosity=1e-3, density=1.0) -> (node_gds, edge_gds):
	''' 
	G: graph
	dG: non-divergence-free boundary (inlets/outlets)

	viscosity & density in SI units / 1000
	''' 
	pressure = node_gds(G)
	velocity = edge_gds(G)

	def pressure_f(t, y):
		div1 = pressure.div(velocity.y/velocity.dt - velocity.advect())
		div2 = pressure.laplacian(velocity.div()) * viscosity/density
		return div1 + div2 - pressure.laplacian(y)

	def velocity_f(t, y):
		return -velocity.advect() - pressure.grad()/density + velocity.laplacian() * viscosity/density

	pressure.set_evolution(cost=pressure_f)
	velocity.set_evolution(dydt=velocity_f)

	return pressure, velocity

def compressible_flow(G: nx.Graph, viscosity=1.0) -> (node_gds, edge_gds):
	raise Exception('Not implemented')

def lagrangian_tracer(velocity: edge_gds, inlets: List[Node]) -> node_gds:
	''' Passive tracer ''' 
	tracer = node_gds(velocity.G)
	tracer.set_evolution(dydt=lambda t, y: -tracer.advect(velocity))
	tracer.set_constraints(dirichlet={i: 1.0 for i in inlets})
	return tracer

''' Systems ''' 

def fluid_on_grid():
	G = grid_graph(10, 10)
	i, o = (3,3), (6,6)
	dG = nx.Graph()
	dG.add_nodes_from([i, o])
	pressure, velocity = incompressible_flow(G, dG)
	pressure.set_constraints(dirichlet={i: 10.0, o: -10.0})
	tracer = lagrangian_tracer(velocity, [i])
	return pressure, velocity, tracer

def fluid_on_circle():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	pressure, velocity = incompressible_flow(G)
	pressure.set_constraints(dirichlet={0: 1.0, n-1: -1.0})
	return pressure, velocity

def differential_inlets():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4,5,6])
	G.add_edges_from([(1,2),(2,3),(4,3),(2,5),(3,5),(5,6)])
	dG = nx.Graph()
	dG.add_nodes_from([1, 4, 6])
	pressure, velocity = incompressible_flow(G, dG)
	pressure.set_constraints(dirichlet={1: 1.0, 4: 0.5, 6: -1.0})
	return pressure, velocity

def poiseuille(m=14, n=21, gradP: float=1.0):
	''' Pressure-driven flow by gradient across dG_L -> dG_R ''' 
	assert n % 2 == 1
	# G = lattice45(m, n)
	G = nx.triangular_lattice_graph(m, n)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	# dG_L.remove_nodes_from([(1, j) for j in range(m)])
	# dG_R.remove_nodes_from([(n-2, j) for j in range(m)])
	dG_L.remove_nodes_from([(0, 2*j+1) for j in range(int(m/2))])
	dG_R.remove_nodes_from([(int(n/2), 2*j+1) for j in range(int(m/2))])
	pressure, velocity = incompressible_flow(G, nx.compose_all([dG_L, dG_R]))
	def pressure_bc(x):
		if x in dG_L.nodes:
			return gradP/2
		elif x in dG_R.nodes:
			return -gradP/2
		return None
	pressure.set_constraints(dirichlet=pressure_bc)
	velocity.set_constraints(dirichlet=no_slip(nx.compose_all([dG_T, dG_B])))
	return pressure, velocity

def poiseuille_asymmetric(m=12, n=24, gradP: float=1.0):
	''' Poiseuille flow with a boundary asymmetry '''
	G = grid_graph(m, n)
	k = 6
	blockage = [
		(k, m-1), (k, m-2),
		(k, m-2), (k, m-3),
		(k, m-3), (k+1, m-3),
		(k+1, m-3), (k+1, m-2),
		(k+1, m-2), (k+1, m-1),
	]
	nbs = nx.node_boundary(G, blockage) | {(k-1, m-4), (k+2, m-4)}
	G.remove_nodes_from(blockage)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	dG_T.add_nodes_from(nbs)
	dG_T.add_edges_from(nx.edge_boundary(G, nbs, nbs))
	pressure, velocity = incompressible_flow(G, nx.compose_all([dG_L, dG_R]))
	def pressure_values(x):
		if x[0] == 0: return gradP/2
		if x[0] == n-1: return -gradP/2
		return None
	pressure.set_constraints(dirichlet=pressure_values)
	velocity.set_constraints(dirichlet=no_slip(nx.compose_all([dG_T, dG_B])))
	return pressure, velocity

def couette(G: nx.Graph, dG_l: nx.Graph, dG_w: nx.Graph, v_l=1.0):
	''' Drag-induced flow by velocity on dG_l with no-slip on dG_w ''' 
	pressure, velocity = incompressible_flow(G)
	wall = noslip(dG_w)
	lid_edges = set(dG_l.edges())
	def bc(t, e):
		v = wall(e)
		if v is None:
			if e in lid_edges:
				return v_l
			elif (e[1], e[0]) in lid_edges:
				return -v_l
		return None
	velocity.set_constraints(dirichlet=bc)
	return pressure, velocity

def von_karman(m=12, n=30, gradP=10.0):
	G = grid_graph(m, n)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	j, k = 6, int(m/2)
	obstacle = [ # Introduce occlusion
		(j, k), 
		(j+1, k-1), (j+1, k), 
		(j+2, k-1),
	]
	G.remove_nodes_from(obstacle)
	pressure, velocity = incompressible_flow(G, nx.compose_all([dG_L, dG_R]))
	def pressure_values(x):
		if x[0] == 0: return gradP/2
		if x[0] == n-1: return -gradP/2
		return None
	pressure.set_constraints(dirichlet=pressure_values)
	velocity.set_constraints(dirichlet=no_slip(nx.compose_all([dG_L, dG_T, dG_B])))
	tracer = lagrangian_tracer(velocity, [(0, 2*i+1) for i in range(int(m/2))])
	return pressure, velocity, tracer

def random_graph():
	set_seed(1001)
	n = 30
	eps = 0.3
	G = nx.random_geometric_graph(n, eps)
	pressure, velocity = incompressible_flow(G)
	pressure.set_constraints(dirichlet=dict_fun({4: 1.0, 21: -1.0}))
	return pressure, velocity

def test1():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4])
	G.add_edges_from([(1,2),(2,3),(3,4)])
	pressure, velocity = incompressible_flow(G)
	p_vals = {}
	v_vals = {(1, 2): 1.0, (2, 3): 1.0}
	pressure.set_constraints(dirichlet=dict_fun(p_vals))
	velocity.set_constraints(dirichlet=dict_fun(v_vals))
	return pressure, velocity

def test2():
	n = 20
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	pressure, velocity = incompressible_flow(G, nx.Graph(), viscosity=0.)
	velocity.set_initial(y0=dict_fun({(2,3): 1.0, (3,4): 1.0}, def_val=0.))
	return pressure, velocity

def lid_driven_cavity(m=15, n=25, v=1.0):
	G = grid_graph(m, n)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	cavity = nx.compose_all([dG_L, dG_R, dG_B])
	lid = dG_T
	pressure, velocity = incompressible_flow(G, nx.Graph())
	velocity.set_constraints(dirichlet=combine_bc([no_slip(cavity), const_velocity(lid, v)]))
	return pressure, velocity

if __name__ == '__main__':
	''' Solve ''' 

	p, v = poiseuille(gradP=10.0)
	# p, v = poiseuille_asymmetric(gradP=10.0)
	# p, v = lid_driven_cavity(v=10.)
	# p, v, t = fluid_on_grid()
	# p, v = differential_inlets()
	# p, v, t = von_karman(n=50, gradP=20)
	# p, v = test2()

	d = v.project(GraphDomain.nodes, lambda v: v.div()) # divergence of velocity
	a = v.project(GraphDomain.edges, lambda v: v.advect()) # advective strength
	# f = v.project(GraphDomain.nodes, lambda v: v.influx()) # mass flux through nodes; assumes divergence-free flow

	sys = couple({
		'pressure': p,
		'velocity': v,
		'divergence': d,
		# 'mass flux': f,
		'advection': a,
		# 'tracer': t,
		# 'grad': grad,
	})

	''' Save to disk ''' 
	# sys.solve_to_disk(20, 1e-2, 'poiseuille')

	''' Load from disk ''' 
	# sys = System.from_disk('von_karman')
	# p, v, d, a = sys.observables['pressure'], sys.observables['velocity'], sys.observables['divergence'], sys.observables['advection']

	canvas = [
		[[[p, v]], [[f]], [[d]]], [[[a]]]
	]

	renderer = LiveRenderer(sys, canvas, node_palette=cc.rainbow, node_rng=(-1,1), node_size=0.03)
	renderer.start()
