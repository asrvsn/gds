import networkx as nx
import numpy as np
import pdb
from itertools import count
import colorcet as cc
import random

from gds.types import *
import gds

''' Definitions ''' 

def incompressible_ns_flow(G: nx.Graph, viscosity=1e-3, density=1.0, v_free=[], **kwargs) -> (gds.node_gds, gds.edge_gds):
	pressure = gds.node_gds(G, **kwargs)
	velocity = gds.edge_gds(G, **kwargs)
	v_free = np.array([pressure.X[x] for x in set(v_free)], dtype=np.intp)
	min_step = 1e-3

	def pressure_f(t, y):
		dt = max(min_step, velocity.dt)
		lhs = velocity.div(velocity.y/dt - velocity.advect()) + pressure.laplacian(velocity.div()) * viscosity/density
		lhs[v_free] = 0.
		lhs -= pressure.laplacian(y)/density 
		return lhs

	def velocity_f(t, y):
		return -velocity.advect() - pressure.grad()/density + velocity.laplacian() * viscosity/density

	pressure.set_evolution(lhs=pressure_f)
	velocity.set_evolution(dydt=velocity_f)

	return velocity, pressure

def incompressible_ns_flow_projected(G: nx.Graph, viscosity=1e-3, density=1.0, v_free=[], body_force=None, **kwargs) -> gds.edge_gds:
	"""
	N-S equations with pressure eliminated.
	"""
	velocity = gds.edge_gds(G, **kwargs)

	nodes = gds.node_gds(G, **kwargs)
	v_free = np.array([nodes.X[x] for x in set(v_free)], dtype=np.intp)

	def velocity_f(t, y):
		rhs = -velocity.advect() + velocity.laplacian() * viscosity/density
		if body_force != None:
			rhs += body_force(t, y)
		return rhs
		# return velocity.leray_project(rhs, v_free)

	velocity.set_evolution(dydt=velocity_f)

	return velocity

def compressible_flow(G: nx.Graph, viscosity=1.0) -> (gds.node_gds, gds.edge_gds):
	raise Exception('Not implemented')

def incompressible_stokes_flow(G: nx.Graph, viscosity=1e-3, density=1.0, inlets=[], outlets=[], **kwargs) -> (gds.node_gds, gds.edge_gds):
	pressure = gds.node_gds(G, **kwargs)
	velocity = gds.edge_gds(G, **kwargs)
	non_div_free = np.array([pressure.X[x] for x in set(inlets) | set(outlets)], dtype=np.intp)
	min_step = 1e-3

	def pressure_f(t, y):
		dt = max(min_step, velocity.dt)
		lhs = velocity.div(velocity.y/dt) + pressure.laplacian(velocity.div()) * viscosity/density
		lhs[non_div_free] = 0.
		lhs -= pressure.laplacian(y)/density 
		return lhs

	def velocity_f(t, y):
		return - pressure.grad()/density + velocity.laplacian() * viscosity/density

	pressure.set_evolution(lhs=pressure_f)
	velocity.set_evolution(dydt=velocity_f)

	return velocity, pressure


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
	velocity, pressure = incompressible_ns_flow(G, dG)
	pressure.set_constraints(dirichlet={i: 10.0, o: -10.0})
	tracer = lagrangian_tracer(velocity, [i])
	return velocity, pressure, tracer

def fluid_on_circle():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	velocity, pressure = incompressible_ns_flow(G)
	pressure.set_constraints(dirichlet={0: 1.0, n-1: -1.0})
	return velocity, pressure

def differential_inlets():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4,5,6])
	G.add_edges_from([(1,2),(2,3),(4,3),(2,5),(3,5),(5,6)])
	velocity, pressure = incompressible_ns_flow(G, inlets=[1,4], outlets=[6], viscosity=1e-2,)
	def vel_f(t, e):
		# omega = 2*np.pi
		if e == (1, 2):
			return 1.
			# return np.sin(omega*t)
		# elif e == (3, 4):
		# 	return 1.
			# return -np.cos(omega*t)
	velocity.set_constraints(dirichlet=vel_f)
	velocity.set_initial(y0=lambda e: -1.0 if e == (2, 3) else 0)
	pressure.set_constraints(dirichlet={6: 0.0})
	return velocity, pressure

def differential_outlets():
	G = nx.Graph()
	G.add_nodes_from(list(range(8)))
	G.add_edges_from([
		(0, 1), 
		(1, 2), (2, 4), (4, 6),
		(1, 3), (3, 5), (5, 7),
		(4, 5)
	])
	velocity, pressure = incompressible_ns_flow(G, inlets=[0], outlets=[6, 7], viscosity=0.01)
	def positive_outlets(vel):
		for outlet in [(4, 6), (5, 7)]:
			vel[velocity.X[outlet]] = max(0, vel[velocity.X[outlet]])
		return vel
	velocity.set_constraints(dirichlet={(0, 1): 1.0}, project=positive_outlets)
	pressure.set_constraints(dirichlet={6: 2.0, 7: 1.98})
	return velocity, pressure

def box_inlets():
	G = nx.Graph()
	G.add_nodes_from([0,1,2,3,4,5,6,7])
	G.add_edges_from([(1,2),(2,3),(2,4),(3,5),(4,5),(5,6),(0,3),(4,7)])
	velocity, pressure = incompressible_ns_flow(G, inlets=[1,0], outlets=[6,7], viscosity=1.)
	def is_vortex():
		v_23, v_35, v_45, v_24 = velocity((2,3)), velocity((3,5)), velocity((4,5)), velocity((2,4))
		ret = np.sign(v_23) == np.sign(v_35)
		ret &= np.sign(v_24) == np.sign(v_45)
		ret &= np.sign(v_23) == -np.sign(v_24)
		ret &= all([np.abs(v) > 0.1 for v in [v_23, v_35, v_45, v_24]])
		return ret
	def vel_f(t, e):
		if e == (1, 2):
			return 0 if is_vortex() else np.sin(t) 
		if e == (0, 3):
			return 0 if is_vortex() else np.sin(t + np.pi/2)
		if e == (5,6):
			return 0 if is_vortex() else -np.sin(t + np.pi)
		if e == (4, 7):
			return 0 if is_vortex() else -np.sin(t + 3*np.pi/2)
	velocity.set_constraints(dirichlet=vel_f)
	# velocity.set_initial(y0=gds.utils.dict_fun({(2,4): 1.0, (4,5): 1.0, (2,3): -1.0, (3,5): -1.0}))
	# pressure.set_constraints(dirichlet={3: 1.0, 4: -1.0})
	return velocity, pressure

def vortex_transfer(viscosity=1e-3):
	G = nx.Graph()
	G.add_nodes_from(list(range(8)))
	outer = set({
		(0,5),(1,5),(1,6),(2,6),(2,7),(3,7),(3,4),(0,4),
	})
	diagonals = set({
		(4,7),(4,5),(5,6),(6,7)
	})
	G.add_edges_from(outer)
	G.add_edges_from(diagonals)
	negated = set({
		(0,4),(1,5),(2,6),(3,7),
	})
	def v_field(e):
		ret = 0
		ret = 1.0 if e in outer else 0
		ret *= -1.0 if e in negated else 1.0
		return ret
	velocity, pressure = incompressible_ns_flow(G, viscosity=viscosity)
	velocity.set_initial(y0=v_field)
	pressure.set_constraints(dirichlet={0: 0.}) # Pressure reference
	return velocity, pressure


def fluid_test(velocity, pressure=None):
	obs = {
		'velocity': velocity,
		'divergence': velocity.project(gds.GraphDomain.nodes, lambda v: v.div()),
		# 'vorticity': velocity.project(gds.GraphDomain.faces, lambda v: v.curl()),
		# 'advective': velocity.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		# 'leray projection': velocity.project(gds.GraphDomain.edges, lambda v: v.leray_project()),
		'L1': velocity.project(PointObservable, lambda v: np.abs(v.y).sum()),
		'L2': velocity.project(PointObservable, lambda v: np.dot(v.y, v.y)),
	}
	if pressure != None:
		obs['pressure'] = pressure
		# obs['pressure_grad'] = pressure.project(gds.GraphDomain.edges, lambda p: -p.grad())
	sys = gds.couple(obs)
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 4), edge_max=0.6, dynamic_ranges=True)


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
	velocity, pressure = incompressible_ns_flow(G, viscosity=100., density=1.0, inlets=l.nodes, outlets=r.nodes, w_key='w')
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
	return velocity, pressure


def random_graph():
	set_seed(1001)
	n = 30
	eps = 0.3
	G = nx.random_geometric_graph(n, eps)
	velocity, pressure = incompressible_ns_flow(G)
	pressure.set_constraints(dirichlet=dict_fun({4: 1.0, 21: -1.0}))
	return velocity, pressure

def test1():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4])
	G.add_edges_from([(1,2),(2,3),(3,4)])
	velocity, pressure = incompressible_ns_flow(G)
	p_vals = {}
	v_vals = {(1, 2): 1.0, (2, 3): 1.0}
	pressure.set_constraints(dirichlet=dict_fun(p_vals))
	velocity.set_constraints(dirichlet=dict_fun(v_vals))
	return velocity, pressure

def test2():
	n = 20
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	velocity, pressure = incompressible_ns_flow(G, viscosity=0.)
	velocity.set_initial(y0=dict_fun({(2,3): 1.0, (3,4): 1.0}, def_val=0.))
	return velocity, pressure

def euler1():
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
	velocity, pressure = incompressible_ns_flow(G, viscosity=0.)
	velocity.set_initial(y0=v_field)
	return velocity, pressure

def euler2():
	v_field = {
		(1,2): 2, (2,3): 1, (3,4): 2, (1,4): -3,
		(5,6): 2, (6,7): 3, (7,8): 2, (5,8): -1,
		(1,5): 1, (2,6): 1, (3,7): -1, (4,8): -1,
	}
	G = gds.flat_cube()
	velocity, pressure = incompressible_ns_flow(G, viscosity=0.)
	velocity.set_initial(y0=lambda e: v_field[e])
	return velocity, pressure

def euler3():
	G = nx.Graph()
	v_field = {
		(1,2): 4, (2,3): 1, (3,4): 4, (4,5): 1, (5,6): 4, (6,7): 1, (7,8): 4, (1,8): -1,
		(9,10): 1, (10,11): 4, (11,12): 1, (12,13): 4, (13,14): 1, (14,15): 4, (15,16): 1, (9, 16): -4,
		(1,9): -3, (2,10): 3, (3,11): -3, (4,12): 3, (5,13): -3, (6,14): 3, (7,15): -3, (8,16): 3, 
	}
	G.add_edges_from(v_field.keys())
	velocity, pressure = incompressible_ns_flow(G, viscosity=0.)
	velocity.set_initial(y0=lambda e: v_field[e])
	return velocity, pressure

def leray1():
	v_field = {
		(1,2): 2, (2,3): 1, (3,4): 2, (1,4): -3,
		(5,6): 2, (6,7): 3, (7,8): 2, (5,8): -1,
		(1,5): 1, (2,6): 1, (3,7): -1, (4,8): -1,
	}
	G = gds.flat_cube()
	velocity = incompressible_ns_flow_projected(G, viscosity=0.)
	velocity.set_initial(y0=lambda e: v_field[e])
	return velocity

def advection_test():
	v_field = {
		(1,2): 2, (2,3): 1, (3,4): 2, (1,4): -3,
		(5,6): 2, (6,7): 3, (7,8): 2, (5,8): -1,
		(1,5): 1, (2,6): 1, (3,7): -1, (4,8): -1,
	}
	G = gds.flat_cube()
	velocity = incompressible_ns_flow_projected(G, viscosity=0.)
	velocity.set_initial(y0=lambda e: v_field[e])

	sys = gds.couple({
		'velocity': velocity,
		'divergence': velocity.project(gds.GraphDomain.nodes, lambda v: v.div()),
		# 'vorticity': velocity.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'advective': velocity.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		'L1': velocity.project(PointObservable, lambda v: np.abs(v.y).sum()),
		'L2': velocity.project(PointObservable, lambda v: np.dot(v.y, v.y)),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True)


if __name__ == '__main__':
	# fluid_test(*fluid_on_grid())
	# fluid_test(*euler3())
	fluid_test(leray1())
	# couette_comp()

