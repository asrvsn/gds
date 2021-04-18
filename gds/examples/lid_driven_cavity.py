'''
Lid-driven cavity (2D drag-induced flow in a no-slip box)
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
from .fluid import incompressible_ns_flow

''' Systems ''' 

def tri_lid_driven_cavity():
	m=18
	n=21
	v=10.0
	# G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	G, (l, r, t, b) = gds.triangular_lattice(m, n*2, with_boundaries=True)
	t.remove_nodes_from([(0, m), (1, m), (n-1, m), (n, m)])
	pressure, velocity = incompressible_ns_flow(G, viscosity=200., density=0.1)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.const_edge_bc(t, v),
		gds.zero_edge_bc(b),
		gds.zero_edge_bc(l),
		gds.zero_edge_bc(r),
	))
	pressure.set_constraints(dirichlet={(0, 0): 0.}) # Pressure reference
	return pressure, velocity

def sq_lid_driven_cavity():
	m=18
	n=21
	v=10.0
	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	t.remove_nodes_from([(0, m-1), (1, m-1), (n-1, m-1), (n, m-1)])
	pressure, velocity = incompressible_ns_flow(G, viscosity=200., density=0.1)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.const_edge_bc(t, v),
		gds.zero_edge_bc(b),
		gds.zero_edge_bc(l),
		gds.zero_edge_bc(r),
	))
	pressure.set_constraints(dirichlet={(0, 0): 0.}) # Pressure reference
	return pressure, velocity

def hex_lid_driven_cavity():
	m=18
	n=21
	v=10.0
	G, (l, r, t, b) = gds.hexagonal_lattice(m, n, with_boundaries=True)
	t.remove_nodes_from([(0, m*2), (1, m*2), (0, m*2+1), (1, m*2+1), (n-1, 2*m), (n, 2*m), (n-1, 2*m+1), (n, 2*m+1)])
	pressure, velocity = incompressible_ns_flow(G, viscosity=200., density=0.1)
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

''' Testing functions ''' 

def render():
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
	render()