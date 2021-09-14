'''
Poiseuille (pressure-induced) flow
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
from .fluid import *

''' Systems ''' 

def tri_poiseuille(viscosity, density):
	m=14 
	n=58 
	gradP=1.0
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	# G.remove_edges_from(l.edges())
	# G.remove_edges_from(r.edges())
	# G.remove_edges_from(G.subgraph([(0, 2*i+1) for i in range(m//2)] + [(1, 2*i) for i in range(m//2+1)]).edges())
	# G.remove_edges_from(G.subgraph([(n//2-1, 2*i+1) for i in range(m//2)] + [(n//2, 2*i) for i in range(m//2+1)]).edges())
	v_free_l = set(l.nodes()) - (set(t.nodes()) | set(b.nodes()))
	# v_free_l_int = set((n[0]+1, n[1]) for n in v_free_l)
	v_free_r = set(r.nodes()) - (set(t.nodes()) | set(b.nodes()))
	# v_free_r_int = set((n[0]-1, n[1]) for n in v_free_r)
	v_free = v_free_l | v_free_r
	velocity, pressure = navier_stokes(G, viscosity=viscosity, density=density, v_free=v_free)
	e_free = np.array([velocity.X[(n, (n[0]+1, n[1]))] for n in v_free_l] + [velocity.X[((n[0]-1, n[1]), n)] for n in v_free_r], dtype=np.intp)
	e_free_mask = np.ones(velocity.ndim)
	e_free_mask[e_free] = 0
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes},
		{(n[0]+1, n[1]): gradP/2 for n in l.nodes},
		{n: -gradP/2 for n in r.nodes},
		{(n[0]-1, n[1]): -gradP/2 for n in r.nodes},
	))
	local_state = {'t': None, 'div': None}
	def free_boundaries(t, e):
		if local_state['t'] != t:
			local_state['div'] = velocity.div(velocity.y*e_free_mask)
		if e[0][1] == e[1][1] and e[0][0] + 1 == e[1][0]:
			if e[0] in v_free:
				return local_state['div'][pressure.X[e[1]]]
			elif e[1] in v_free:
				return -local_state['div'][pressure.X[e[0]]]
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
		free_boundaries
	))
	return velocity, pressure

def sq_poiseuille(viscosity, density):
	m=14 
	n=28 
	gradP=6.0

	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	v_free_l = set(l.nodes()) - (set(t.nodes()) | set(b.nodes()))
	v_free_r = set(r.nodes()) - (set(t.nodes()) | set(b.nodes()))
	v_free = v_free_l | v_free_r

	velocity, pressure = navier_stokes(G, viscosity=viscosity, density=density, v_free=v_free)

	e_free = np.array([velocity.X[(n, (n[0]+1, n[1]))] for n in v_free_l] + [velocity.X[((n[0]-1, n[1]), n)] for n in v_free_r], dtype=np.intp)
	e_free_mask = np.ones(velocity.ndim)
	e_free_mask[e_free] = 0

	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes},
		{(n[0]+1, n[1]): gradP/2 for n in l.nodes},
		{n: -gradP/2 for n in r.nodes},
		{(n[0]-1, n[1]): -gradP/2 for n in r.nodes},
	))
	local_state = {'t': None, 'div': None}
	def free_boundaries(t, e):
		if local_state['t'] != t:
			local_state['div'] = velocity.div(velocity.y*e_free_mask)
		if e[0][1] == e[1][1] and e[0][0] + 1 == e[1][0]:
			if e[0] in v_free:
				return local_state['div'][pressure.X[e[1]]]
			elif e[1] in v_free:
				return -local_state['div'][pressure.X[e[0]]]
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
		free_boundaries
	))
	return velocity, pressure

def hex_poiseuille(viscosity, density):
	m=14 
	n=29 
	gradP=1.0

	G, (l, r, t, b) = gds.hexagonal_lattice(m, n, with_boundaries=True)
	v_free_l = set(l.nodes()) - (set(t.nodes()) | set(b.nodes()))
	# v_free_l.remove(min(v_free_l, key=lambda x: x[1]))
	# v_free_l.remove(max(v_free_l, key=lambda x: x[1]))
	v_free_r = set(r.nodes()) - (set(t.nodes()) | set(b.nodes()))
	# v_free_r.remove(min(v_free_r, key=lambda x: x[1]))
	# v_free_r.remove(max(v_free_r, key=lambda x: x[1]))
	v_free = v_free_l | v_free_r
	v_bd_l = set()
	for v in v_free_l:
		u = (-1, v[1])
		G.add_node(u)
		G.add_edge(u, v)
		G.nodes[u]['pos'] = (G.nodes[v]['pos'][0] - 1.0, G.nodes[v]['pos'][1])
		v_bd_l.add(u)
	v_bd_r = set()
	for v in v_free_r:
		u = (n+1, v[1])
		G.add_node(u)
		G.add_edge(v, u)
		G.nodes[u]['pos'] = (G.nodes[v]['pos'][0] + 1.0, G.nodes[v]['pos'][1])
		v_bd_r.add(u)
	v_bd = v_bd_l | v_bd_r

	velocity, pressure = navier_stokes(G, viscosity=viscosity, density=density, v_free=v_bd | v_free)

	e_free_mask = np.array([1 if len(set(velocity.iX[i]) - v_bd)==2 else 0 for i in range(velocity.ndim)])

	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in v_free_l | v_bd_l},
		{n: -gradP/2 for n in v_free_r | v_bd_r},
	))
	local_state = {'t': None, 'div': None}
	def free_boundaries(t, e):
		if local_state['t'] != t:
			local_state['div'] = velocity.div(velocity.y*e_free_mask)
			local_state['t'] = t
		if e[1] in v_bd_l:
			return -local_state['div'][pressure.X[e[0]]]
		elif e[1] in v_bd_r:
			return -local_state['div'][pressure.X[e[0]]]
		# elif (e[0][1] == e[1][1] == 0) and (e[0][0] == e[1][0] - 1):
		# 	return 0
		# elif (e[0][1] == e[1][1] == n) and (e[0][0] == e[1][0] - 1):
		# 	return 0
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		free_boundaries,
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
	))
	return velocity, pressure

def voronoi_poiseuille(viscosity, density):
	np.random.seed(401)
	gradP=1.0
	n_boundary = 10
	n_interior = 100

	G, (l, r, t, b) = gds.voronoi_lattice(n_boundary, n_interior, with_boundaries=True, eps=0.07)
	# tb = set(t.nodes()) | set(b.nodes())
	v_free_l, v_free_r = set(l.nodes()), set(r.nodes())
	v_free = v_free_l | v_free_r
	v_bd_l = set()
	# for v in v_free_l:
	# 	u = v - 1j
	# 	G.add_node(u)
	# 	G.add_edge(u, v)
	# 	G.nodes[u]['pos'] = (G.nodes[v]['pos'][0] - 0.1, G.nodes[v]['pos'][1])
	# 	v_bd_l.add(u)
	v_bd_r = set()
	# for v in v_free_r:
	# 	u = v - 1j
	# 	G.add_node(u)
	# 	G.add_edge(v, u)
	# 	G.nodes[u]['pos'] = (G.nodes[v]['pos'][0] + 0.1, G.nodes[v]['pos'][1])
	# 	v_bd_r.add(u)
	v_bd = v_bd_l | v_bd_r

	velocity, pressure = navier_stokes(G, viscosity=viscosity, density=density, v_free=v_bd | v_free)
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in v_free_l | v_bd_l},
		{n: -gradP/2 for n in v_free_r | v_bd_r},
	))
	# e_free_mask = np.array([1 if len(set(velocity.iX[i]) - v_bd)==2 else 0 for i in range(velocity.ndim)])
	# local_state = {'t': None, 'div': None}
	# def free_boundaries(t, e):
	# 	if local_state['t'] != t:
	# 		local_state['div'] = velocity.div(velocity.y*e_free_mask)
	# 		local_state['t'] = t
	# 	if e[1] in v_bd_l:
	# 		return -np.clip(local_state['div'][pressure.X[e[0]]], 0, None)
	# 	elif e[1] in v_bd_r:
	# 		return -np.clip(local_state['div'][pressure.X[e[0]]], None, 0)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		# free_boundaries,
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
	))
	return velocity, pressure

def poiseuille_flow():
	''' API example ''' 
	G, (G_L, G_R, G_T, G_B) = gds.square_lattice(m, n, with_boundaries=True)

	velocity = Field(GraphDomain.Edges, G)
	velocity.set_dirichlet_boundary(set(G_T.edges()) | set(G_B.edges()), 0.)
	velocity.set_free_boundary(set(G_L.edges()) | set(G_R.edges()))

	pressure = Field(GraphDomain.Nodes, G)
	pressure.set_dirichlet_boundary(set(G_L.edges()), 1.)
	pressure.set_dirichlet_boundary(set(G_R.edges()), -1.)
	pressure.set_free_boundary(set(G_T.edges()) | set(G_B.edges()))

	dt = 1e-3
	velocity.set_evolution(
		dydt = -advect(velocity, velocity) - grad(pressure) / density + laplacian(velocity) * viscosity / density
	)
	pressure.set_evolution(
		lhs = div(velocity / dt - advect(velocity, velocity)) - laplacian(pressure) / density + laplacian(div(velocity)) * viscosity / density
	)

def sq_poiseuille_projected(viscosity, density):
	m=14 
	n=28 
	gradP=6.0

	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	v_free_l = set(l.nodes()) - (set(t.nodes()) | set(b.nodes()))
	v_free_r = set(r.nodes()) - (set(t.nodes()) | set(b.nodes()))
	v_free = v_free_l | v_free_r

	velocity = navier_stokes_projected(G, viscosity=viscosity, density=density, body_force=lambda t, y: gradP*np.ones_like(y), v_free=v_free)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
	))
	return velocity


''' Testing functions ''' 

def render():
	viscosity, density = 1., 1e-2
	# p, v = voronoi_poiseuille()
	# p, v = sq_poiseuille()
	p1, v1 = sq_poiseuille(viscosity, density)
	p2, v2 = tri_poiseuille(viscosity, density)
	p3, v3 = hex_poiseuille(viscosity, density)

	# v = v3
	sys = gds.couple({
		# 'velocity': v,
		'velocity_sq': v1,
		'velocity_tri': v2,
		'velocity_hex': v3,
		# 'pressure': p,
		'pressure_sq': p1,
		'pressure_tri': p2,
		'pressure_hex': p3,
		# 'vorticity': v.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'vorticity_sq': v1.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'vorticity_tri': v2.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'vorticity_hex': v3.project(gds.GraphDomain.faces, lambda v: v.curl()),
		# 'div_sq': v1.project(gds.GraphDomain.nodes, lambda v: v.div()),
		# 'div_tri': v2.project(gds.GraphDomain.nodes, lambda v: v.div()),
		# 'div_hex': v3.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'mass_flux_sq': v1.project(GraphDomain.edges, lambda v: viscosity * v.laplacian() - p1.grad()), 
		'mass_flux_tri': v2.project(GraphDomain.edges, lambda v: viscosity * v.laplacian() - p2.grad()), 
		'mass_flux_hex': v3.project(GraphDomain.edges, lambda v: viscosity * v.laplacian() - p3.grad()), 
		# 'divergence': v.project(gds.GraphDomain.nodes, lambda v: v.div()),
		# 'laplacian_sq': v1.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		# 'laplacian_tri': v2.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		# 'laplacian_hex': v3.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		# 'dd*': v.project(gds.GraphDomain.edges, lambda v: v.dd_()),
		# 'd*d': v.project(gds.GraphDomain.edges, lambda v: v.d_d()),
		# 'energy': v.project(PointObservable, lambda v: (v.y ** 2).sum()),
		# 'momentum': v.project(PointObservable, lambda v: np.abs(v.y).sum()),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True, plot_width=900, node_size=0.04)

def dump():
	p1, v1 = sq_poiseuille()
	p2, v2 = tri_poiseuille()
	p3, v3 = hex_poiseuille()

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
		'diffusion_square': v1.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		'diffusion_tri': v2.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		'diffusion_hex': v3.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		'advection_square': v1.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		'advection_tri': v2.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		'advection_hex': v3.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		'divergence_square': v1.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'divergence_tri': v2.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'divergence_hex': v3.project(gds.GraphDomain.nodes, lambda v: v.div()),
	})

	sys.solve_to_disk(5.0, 0.01, 'poiseuille')

if __name__ == '__main__':
	# render()
	# dump()
	fluid_test(*sq_poiseuille(50, 1))