import networkx as nx
import numpy as np
import pdb
from itertools import count
import colorcet as cc
import random
import cvxpy as cp

from gds.types import *
import gds

''' Definitions ''' 

def incompressible_flow(G: nx.Graph, viscosity=1e-3, density=1.0, inlets=[], outlets=[], **kwargs) -> (gds.node_gds, gds.edge_gds):
	''' 
	G: graph
	viscosity & density in SI units / 1000
	''' 
	pressure = gds.node_gds(G, **kwargs)
	velocity = gds.edge_gds(G, **kwargs)
	non_div_free = np.array([pressure.X[x] for x in set(inlets) | set(outlets)], dtype=np.intp)
	min_step = 1e-4

	def pressure_f(t, y):
		dt = max(min_step, velocity.dt)
		lhs = velocity.div(velocity.y/dt - velocity.advect()) + pressure.laplacian(velocity.div()) * viscosity/density
		lhs[non_div_free] = 0.
		lhs -= pressure.laplacian(y)/density 
		return lhs

	def velocity_f(t, y):
		return -velocity.advect() - pressure.grad()/density + velocity.laplacian() * viscosity/density

	pressure.set_evolution(lhs=pressure_f)
	velocity.set_evolution(dydt=velocity_f)

	return pressure, velocity

def compressible_flow(G: nx.Graph, viscosity=1.0) -> (gds.node_gds, gds.edge_gds):
	raise Exception('Not implemented')

def lagrangian_tracer(velocity: gds.edge_gds, inlets: List[Node], alpha=1.0) -> gds.node_gds:
	''' Passive tracer ''' 
	tracer = gds.node_gds(velocity.G)
	tracer.set_evolution(dydt=lambda t, y: -alpha*tracer.advect(velocity))
	tracer.set_constraints(dirichlet={i: 1.0 for i in inlets})
	return tracer

''' Systems ''' 

def fluid_on_grid():
	G = gds.square_lattice(10, 10)
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
	pressure, velocity = incompressible_flow(G, inlets=[1,4], outlets=[6], viscosity=1.)
	velocity.set_constraints(dirichlet={
		(1, 2): 100.0,
		(3, 4): -1
	})
	pressure.set_constraints(dirichlet={6: 0.0})
	return pressure, velocity

def poiseuille():
	''' Pressure-driven flow by gradient across dG_L -> dG_R ''' 
	m=14 
	n=31 
	gradP=1.0
	# assert n % 2 == 1
	# G = lattice45(m, n)
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	pressure, velocity = incompressible_flow(G, viscosity=1., density=1e-3, inlets=l.nodes, outlets=r.nodes)
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes},
		{n: -gradP/2 for n in r.nodes}
	))
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
	))
	return pressure, velocity

def backward_step():
	''' Poiseuille flow with a boundary asymmetry '''
	m=22 
	n=67 
	step_height=12
	step_width=10
	obstacle=gds.utils.flatten([[(j,i) for i in range(step_height)] for j in range(step_width)])
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	for g in [G, l, r, t, b]:
		g.remove_nodes_from(obstacle)
	for i in range(step_width+1):
		b.add_node((i, step_height))
		if i > 0:
			b.add_edge((i-1, step_height), (i, step_height))
	for j in range(step_height+1):
		b.add_node((step_width, j))
		if i > 0:
			b.add_edge((step_width, j-1), (step_width, j))
	G.remove_edges_from(list(nx.edge_boundary(G, l, l)))
	G.remove_edges_from(list(nx.edge_boundary(G, [(0, 2*i+1) for i in range(m//2)], [(1, 2*i) for i in range(m//2+1)])))
	G.remove_edges_from(list(nx.edge_boundary(G, r, r)))
	G.remove_edges_from(list(nx.edge_boundary(G, [(n//2, 2*i+1) for i in range(m//2)], [(n//2, 2*i) for i in range(m//2+1)])))
	weight = 1.
	nx.set_edge_attributes(G, weight, name='w')

	inlet_v=1.0
	# outlet_v=2*(m - step_height - 2)*inlet_v / (m - 2)
	outlet_p=0.0
	# ref_p=0.0
	# grad_p=100.0
	pressure, velocity = incompressible_flow(G, viscosity=100., density=1.0, inlets=l.nodes, outlets=r.nodes, w_key='w')
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		# {n: grad_p/2 for n in l.nodes},
		# {n: -grad_p/2 for n in r.nodes if n[1] > step_height//2}
		{(n//2+1, j): outlet_p for j in range(n)}
		# {(n//2+1,m): ref_p}
	))
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		{((0, i), (1, i)): inlet_v for i in range(step_height+1, m)},
		# {((n//2, i), (n//2+1, i)): outlet_v for i in range(1, m)},
		# {((n//2-1, 2*i+1), (n//2, 2*i+1)): outlet_v for i in range(0, m//2)},
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
	))
	return pressure, velocity

def lid_driven_cavity():
	''' Drag-induced flow ''' 
	m=18
	n=21
	v=10.0
	# G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	G, (l, r, t, b) = gds.triangular_lattice(m, n*2, with_boundaries=True)
	t.remove_nodes_from([(0, m), (1, m), (n-1, m), (n, m)])
	pressure, velocity = incompressible_flow(G, viscosity=200., density=0.1)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.const_edge_bc(t, v),
		gds.zero_edge_bc(b),
		gds.zero_edge_bc(l),
		gds.zero_edge_bc(r),
	))
	pressure.set_constraints(dirichlet={(0, 0): 0.}) # Pressure reference
	return pressure, velocity

def von_karman():
	m=24 
	n=113 
	gradP=10.0
	inlet_v = 5.0
	outlet_p = 0.0
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	weight = 1.0
	nx.set_edge_attributes(G, weight, name='w')
	j, k = 8, m//2
	# Introduce occlusion
	obstacle = [ 
		(j, k), 
		(j+1, k),
		(j, k+1), 
		(j, k-1),
		# (j-1, k), 
		# (j+1, k+1), 
		# (j+1, k-1),
		# (j, k+2), 
		# (j, k-2), 
	]
	obstacle_boundary = gds.utils.flatten([G.neighbors(n) for n in obstacle])
	obstacle_boundary = list(nx.edge_boundary(G, obstacle_boundary, obstacle_boundary))
	G.remove_nodes_from(obstacle)
	G.remove_edges_from(list(nx.edge_boundary(G, l, l)))
	G.remove_edges_from(list(nx.edge_boundary(G, [(0, 2*i+1) for i in range(m//2)], [(1, 2*i) for i in range(m//2+1)])))
	G.remove_edges_from(list(nx.edge_boundary(G, r, r)))
	G.remove_edges_from(list(nx.edge_boundary(G, [(n//2, 2*i+1) for i in range(m//2)], [(n//2, 2*i) for i in range(m//2+1)])))
	pressure, velocity = incompressible_flow(G, viscosity=1e-4, density=1, inlets=l.nodes, outlets=r.nodes, w_key='w')
	# pressure.set_constraints(dirichlet=gds.combine_bcs(
		# {n: gradP/2 for n in l.nodes},
		# {n: -gradP/2 for n in r.nodes}
		# {(n//2+1, j): outlet_p for j in range(n)}
	# ))
	gradation = np.linspace(-0.5, 0.5, m+1)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		{((0, i), (1, i)): inlet_v + gradation[i] for i in range(1, m)},
		{((n//2, i), (n//2+1, i)): inlet_v - gradation[i] for i in range(1, m)},
		{((n//2-1, 2*i+1), (n//2, 2*i+1)): inlet_v - gradation[2*i+1] for i in range(0, m//2)},
		gds.utils.bidict({e: 0 for e in obstacle_boundary}),
		gds.utils.bidict({e: 0 for e in t.edges}),
		gds.utils.bidict({e: 0 for e in b.edges})

	))
	return pressure, velocity

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

if __name__ == '__main__':
	''' Solve ''' 

	# p, v = poiseuille()
	# p, v = poiseuille_asymmetric(gradP=10.0)
	# p, v = lid_driven_cavity()
	# p, v, t = fluid_on_grid()
	# p, v = differential_inlets()
	# p, v = von_karman()
	p, v = backward_step()

	d = v.project(GraphDomain.nodes, lambda v: v.div()) # divergence of velocity
	a = v.project(GraphDomain.edges, lambda v: -v.advect()) # advective strength
	# f = v.project(GraphDomain.nodes, lambda v: v.influx()) # mass flux through nodes; assumes divergence-free flow
	# m = v.project(GraphDomain.edges, lambda v: v.laplacian()) # momentum diffusion

	sys = gds.couple({
		'pressure': p,
		'velocity': v,
		'divergence': d,
		# 'mass flux': f,
		'advection': a,
		# 'momentum diffusion': m,
		# 'tracer': t,
		# 'grad': grad,
	})

	''' Save to disk ''' 
	# sys.solve_to_disk(20, 1e-2, 'poiseuille')

	''' Load from disk ''' 
	# sys = System.from_disk('von_karman')
	# p, v, d, a = sys.observables['pressure'], sys.observables['velocity'], sys.observables['divergence'], sys.observables['advection']

	canvas = [
		[[[p]], [[v]]],
		[[[a]], [[d]]],
		# [[[a]], [[f]]], 
		# [[[m]]],
	]
	# gds.render(sys, canvas=canvas, node_palette=cc.rainbow, dynamic_ranges=True, node_size=0.04, plot_width=800, plot_height=550, y_rng=(-1.1,0.7))
	gds.render(sys, canvas=canvas, node_palette=cc.rainbow, edge_palette=cc.rainbow, dynamic_ranges=True, node_size=0.03, edge_max=0.3, edge_colors=True, plot_width=800, plot_height=500, y_rng=(-1.1,0.5))


