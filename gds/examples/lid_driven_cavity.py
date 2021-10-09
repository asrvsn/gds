'''
Lid-driven cavity (2D drag-induced flow in a no-slip box)
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
from .fluid import *
import gds.examples.fluid_projected as projected

''' Systems ''' 

def lid_driven_cavity(G, dG, v=10, viscosity=10):
	(l, r, t, b) = dG
	v_free = (set(l.nodes()) & set(t.nodes())) | (set(r.nodes()) & set(t.nodes()))
	print('v_free:', v_free)
	velocity, pressure = navier_stokes(G, viscosity=viscosity, v_free=v_free)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.const_edge_bc(t, v),
		gds.zero_edge_bc(b),
		gds.zero_edge_bc(l),
		gds.zero_edge_bc(r),
	))
	# pressure.set_constraints(dirichlet={(0, 0): 0.}) # Pressure reference
	return velocity, pressure

def tri_lid_driven_cavity_projected():
	m=18
	n=21
	v=10.0
	# G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	G = gds.triangular_lattice(m, n*2)
	# t.remove_nodes_from([(0, m), (1, m), (n-1, m), (n, m)])
	velocity, pressure = projected.navier_stokes(G, viscosity=10., density=1., v_free=[(0, m), (n, m)])
	# velocity.set_constraints(dirichlet=gds.combine_bcs(
	# 	gds.const_edge_bc(t, v),
	# 	gds.zero_edge_bc(b),
	# 	gds.zero_edge_bc(l),
	# 	gds.zero_edge_bc(r),
	# ))
	pressure.set_constraints(dirichlet={
		(0, m): 10,
		(n, m): -10
	})
	return velocity, pressure

def sq_lid_driven_cavity_ivp():
	m=18
	n=21
	v=1.0
	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	velocity, pressure = navier_stokes(G, viscosity=200., density=0.1)
	def walls(e):
		if e in t.edges(): return v
		elif e in b.edges(): return -v
		elif e in r.edges(): return -v
		elif (e[1],e[0]) in r.edges(): return v
		elif e in l.edges(): return v
		elif (e[1],e[0]) in l.edges(): return -v
		return 0
	velocity.set_initial(y0=walls)
	pressure.set_constraints(dirichlet={(0, 0): 0.}) # Pressure reference
	return velocity, pressure

def tri_lid_driven_cavity_ivp():
	m=18
	n=21
	v=1.0
	G, (l, r, t, b) = gds.triangular_lattice(m, n*2, with_boundaries=True)
	t.remove_nodes_from([(0, m), (1, m), (n-1, m), (n, m)])
	velocity, pressure = navier_stokes(G, viscosity=200., density=0.1)
	velocity.set_initial(y0=lambda e: v if e in t.edges else 0)
	return velocity, pressure

''' Testing functions ''' 

def render():
	p1, v1 = sq_lid_driven_cavity_ivp()
	# p1, v1 = tri_lid_driven_cavity_ivp()
	# p1, v1 = sq_lid_driven_cavity()
	# p2, v2 = tri_lid_driven_cavity()
	# p3, v3 = hex_lid_driven_cavity()

	sys = gds.couple({
		'velocity_square': v1,
		# 'velocity_tri': v2,
		# 'velocity_hex': v3,
		# 'pressure_square': p1,
		# 'pressure_tri': p2,
		# 'pressure_hex': p3,
		'vorticity_square': v1.project(gds.GraphDomain.faces, lambda v: v.curl()),
		# 'vorticity_tri': v2.project(gds.GraphDomain.faces, lambda v: v.curl()),
		# 'vorticity_hex': v3.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'kinetic energy': v1.project(PointObservable, lambda v: (v1.y ** 2).sum(), min_rng=0.1),
		'momentum': v1.project(PointObservable, lambda v: np.abs(v1.y).sum(), min_rng=0.1),
		'rotational energy': v1.project(PointObservable, lambda v: (v1.curl() ** 2).sum(), min_rng=0.1),
		'angular momentum': v1.project(PointObservable, lambda v: np.abs(v1.curl()).sum(), min_rng=0.1),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True)

def dump():
	p1, v1 = sq_lid_driven_cavity()
	p2, v2 = tri_lid_driven_cavity()
	p3, v3 = hex_lid_driven_cavity()

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

	sys.solve_to_disk(5.0, 0.01, 'lid_driven_cavity')

if __name__ == '__main__':
	# render()

	# G, dG = gds.triangular_lattice(18, 42, with_boundaries=True)
	G, dG = gds.square_lattice(16, 30, with_boundaries=True)
	# G, dG = gds.hexagonal_lattice(18, 21, with_boundaries=True)

	v, p = lid_driven_cavity(G, dG, viscosity=200)
	fluid_test(v, p, columns=3, edge_palette=cc.bgy)

