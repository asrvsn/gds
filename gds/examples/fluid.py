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
	''' 
	pressure = gds.node_gds(G, **kwargs)
	velocity = gds.edge_gds(G, **kwargs)
	non_div_free = np.array([pressure.X[x] for x in set(inlets) | set(outlets)], dtype=np.intp)
	min_step = 1e-3

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
	pressure, velocity = incompressible_flow(G, inlets=[1,4], outlets=[6], viscosity=1e-2,)
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
	return pressure, velocity

def differential_outlets():
	G = nx.Graph()
	G.add_nodes_from(list(range(8)))
	G.add_edges_from([
		(0, 1), 
		(1, 2), (2, 4), (4, 6),
		(1, 3), (3, 5), (5, 7),
		(4, 5)
	])
	pressure, velocity = incompressible_flow(G, inlets=[0], outlets=[6, 7], viscosity=0.01)
	def positive_outlets(vel):
		for outlet in [(4, 6), (5, 7)]:
			vel[velocity.X[outlet]] = max(0, vel[velocity.X[outlet]])
		return vel
	velocity.set_constraints(dirichlet={(0, 1): 1.0}, project=positive_outlets)
	pressure.set_constraints(dirichlet={6: 2.0, 7: 1.98})
	return pressure, velocity

def box_inlets():
	G = nx.Graph()
	G.add_nodes_from([0,1,2,3,4,5,6,7])
	G.add_edges_from([(1,2),(2,3),(2,4),(3,5),(4,5),(5,6),(0,3),(4,7)])
	pressure, velocity = incompressible_flow(G, inlets=[1,0], outlets=[6,7], viscosity=1.)
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
	return pressure, velocity

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
	pressure, velocity = incompressible_flow(G, viscosity=viscosity)
	velocity.set_initial(y0=v_field)
	pressure.set_constraints(dirichlet={0: 0.}) # Pressure reference
	return pressure, velocity

def sq_couette():
	m, n = 11, 10
	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	faces, outer_face = gds.embedded_faces(G)
	for j in range(m):
		G.add_edge((n-1, j), (0, j))
	aux_faces = [((n-1, j), (0, j), (0, j+1), (n-1, j+1)) for j in range(m-1)]
	G.faces = faces + aux_faces # Hacky
	G.rendered_faces = np.array(range(len(faces)), dtype=np.intp) # Hacky

	pressure, velocity = incompressible_flow(G, viscosity=1., density=1e-2)
	vel = 1.0
	def walls(e):
		if e in t.edges or e == ((0, m-1), (n-1, m-1)):
			return 0
		elif e in b.edges: 
			return vel
		elif e == ((0, 0), (n-1, 0)):
			return -vel
		return None
	velocity.set_constraints(dirichlet=walls)
	return pressure, velocity

def tri_couette():
	m, n = 10, 20
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	faces, outer_face = gds.embedded_faces(G)
	for j in range(m+1):
		G = nx.algorithms.minors.contracted_nodes(G, (0, j), ((n + 1) // 2, j))
	rendered_faces = set()
	r_nodes = set(r.nodes())
	for i, face in enumerate(faces):
		face = list(face)
		modified = False
		for j, node in enumerate(face):
			if node in r_nodes:
				n_l = (0, node[1]) # identified
				face[j] = n_l
				faces[i] = tuple(face)
				modified = True
		if not modified:
			rendered_faces.add(i)
	G.faces = faces
	G.rendered_faces = np.array(sorted(list(rendered_faces)), dtype=np.intp) # Hacky

	pressure, velocity = incompressible_flow(G, viscosity=1., density=1e-2)
	vel = 1.0
	def walls(e):
		if e in b.edges:
			return vel
		elif e == ((0, 0), (n//2-1, 0)):
			return -vel
		elif e in t.edges or e == ((0, m), (n//2-1, m)):
			return 0.0
		return None
	velocity.set_constraints(dirichlet=walls)
	return pressure, velocity

def hex_couette():
	m, n = 6, 12
	G, (l, r, t, b) = gds.hexagonal_lattice(m, n, with_boundaries=True)
	faces, outer_face = gds.embedded_faces(G)
	contractions = {}
	for j in range(1, 2*m+1):
		G = nx.algorithms.minors.contracted_nodes(G, (0, j), (n, j))
		contractions[(n, j)] = (0, j)
	nx.set_node_attributes(G, None, 'contraction')
	rendered_faces = set()
	for i, face in enumerate(faces):
		face = list(face)
		modified = False
		for j, node in enumerate(face):
			if node in contractions:
				n_l = contractions[node] # identified
				face[j] = n_l
				faces[i] = tuple(face)
				modified = True
		if not modified:
			rendered_faces.add(i)
	G.faces = faces
	G.rendered_faces = np.array(sorted(list(rendered_faces)), dtype=np.intp) # Hacky

	pressure, velocity = incompressible_flow(G, viscosity=1., density=1e-2)
	vel = 1.0
	def walls(e):
		if (e[0][1] == e[1][1] == 0) and (e[0][0] == e[1][0] - 1):
			return vel
		elif e in t.edges or e == ((0, 2*m), (n, 2*m+1)):
			return 0.0
		return None
	velocity.set_constraints(dirichlet=walls)
	return pressure, velocity

def fluid_test(pressure, velocity):
	sys = gds.couple({
		'pressure': pressure,
		'velocity': velocity,
		'divergence': velocity.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'vorticity': velocity.project(gds.GraphDomain.faces, lambda v: v.curl()),
	})
	gds.render(sys, edge_max=0.6, dynamic_ranges=True)

def couette_comp():
	p1, v1 = sq_couette()
	p2, v2 = tri_couette()
	p3, v3 = hex_couette()

	sys = gds.couple({
		'velocity_square': v1,
		'velocity_tri': v2,
		'velocity_hex': v3,
		'pressure_square': p1,
		'pressure_tri': p2,
		'pressure_hex': p3,
		'vorticity_square': v1.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'vorticity_tri': v2.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'vorticity_hex': v3.project(gds.GraphDomain.faces, lambda v: v.curl()),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True)

def poiseuille():
	''' Pressure-driven flow by gradient across dG_L -> dG_R ''' 
	m=14 
	n=58 
	gradP=1.0
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	pressure, velocity = incompressible_flow(G, viscosity=1., density=1e-2, inlets=l.nodes, outlets=r.nodes)
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes if not (n in t.nodes or n in b.nodes)},
		{n: -gradP/2 for n in r.nodes if not (n in t.nodes or n in b.nodes)}
	))
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
	))
	return pressure, velocity

def poiseuille_sq():
	''' Pressure-driven flow by gradient across dG_L -> dG_R ''' 
	m=14 
	n=28 
	gradP=1.0
	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	pressure, velocity = incompressible_flow(G, viscosity=1., density=1e-2, inlets=l.nodes, outlets=r.nodes)
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes if not (n in t.nodes or n in b.nodes)},
		{n: -gradP/2 for n in r.nodes if not (n in t.nodes or n in b.nodes)}
	))
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
	))
	return pressure, velocity

def poiseuille_hex():
	''' Pressure-driven flow by gradient across dG_L -> dG_R ''' 
	m=14 
	n=28 
	gradP=1.0
	G, (l, r, t, b) = gds.hexagonal_lattice(m, n, with_boundaries=True)
	pressure, velocity = incompressible_flow(G, viscosity=1., density=1e-2, inlets=l.nodes, outlets=r.nodes)
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes if not (n in t.nodes or n in b.nodes)},
		{n: -gradP/2 for n in r.nodes if not (n in t.nodes or n in b.nodes)}
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

def lid_driven_cavity_sq():
	''' Drag-induced flow ''' 
	m=18
	n=21
	v=10.0
	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	t.remove_nodes_from([(0, m-1), (1, m-1), (n-1, m-1), (n, m-1)])
	pressure, velocity = incompressible_flow(G, viscosity=200., density=0.1)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.const_edge_bc(t, v),
		gds.zero_edge_bc(b),
		gds.zero_edge_bc(l),
		gds.zero_edge_bc(r),
	))
	pressure.set_constraints(dirichlet={(0, 0): 0.}) # Pressure reference
	return pressure, velocity

def lid_driven_cavity_hex():
	''' Drag-induced flow ''' 
	m=18
	n=21
	v=10.0
	G, (l, r, t, b) = gds.hexagonal_lattice(m, n, with_boundaries=True)
	t.remove_nodes_from([(0, m*2), (1, m*2), (0, m*2+1), (1, m*2+1), (n-1, 2*m), (n, 2*m), (n-1, 2*m+1), (n, 2*m+1)])
	pressure, velocity = incompressible_flow(G, viscosity=200., density=0.1)
	def inlet_bc(e):
		if e in t.edges:
			if e[1][1] > e[0][1] and e[0][0] % 2 == 0:
				# Hack to fix hexagonal bcs
				return -v
			return v
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		inlet_bc,
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

def save_poiseuille():
	p1, v1 = poiseuille()
	p2, v2 = poiseuille_sq()
	p3, v3 = poiseuille_hex()

	c1 = v1.project(GraphDomain.faces, lambda v: v.curl())
	c2 = v2.project(GraphDomain.faces, lambda v: v.curl())
	c3 = v3.project(GraphDomain.faces, lambda v: v.curl())

	d1 = v1.project(GraphDomain.nodes, lambda v: v.div())
	d2 = v2.project(GraphDomain.nodes, lambda v: v.div())
	d3 = v3.project(GraphDomain.nodes, lambda v: v.div())

	m1 = v1.project(GraphDomain.edges, lambda v: v.laplacian()) 
	m2 = v2.project(GraphDomain.edges, lambda v: v.laplacian()) 
	m3 = v3.project(GraphDomain.edges, lambda v: v.laplacian()) 

	a1 = v1.project(GraphDomain.edges, lambda v: -v.advect()) 
	a2 = v2.project(GraphDomain.edges, lambda v: -v.advect()) 
	a3 = v3.project(GraphDomain.edges, lambda v: -v.advect()) 

	sys = gds.couple({
		'tri_velocity': v1,
		'tri_pressure': p1,
		'tri_vorticity': c1,
		'tri_divergence': d1,
		'tri_diffusion': m1,
		'tri_advection': a1,
		'sq_velocity': v2,
		'sq_pressure': p2,
		'sq_vorticity': c2,
		'sq_divergence': d2,
		'sq_diffusion': m2,
		'sq_advection': a2,
		'hex_velocity': v3,
		'hex_pressure': p3,
		'hex_vorticity': c3,
		'hex_divergence': d3,
		'hex_diffusion': m3,
		'hex_advection': a3,
		# 'velocity3': v3,
		# 'pressure3': p3,
		# 'divergence': d,
		# 'mass flux': f,
		# 'advection': a,
		# 'momentum diffusion': m,
		# 'tracer': t,
		# 'grad': grad,
		# 'velocity @ viscosity=1e-1': v1,
		# 'velocity @ viscosity=100': v2,
		# 'p1': p1,
		# 'p2': p2,
	})

	''' Save to disk ''' 
	sys.solve_to_disk(5.0, 0.01, 'poiseuille')


if __name__ == '__main__':
	# fluid_test(*hex_couette())
	couette_comp()

