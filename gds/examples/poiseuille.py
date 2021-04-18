'''
Poiseuille (pressure-induced) flow
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
from .fluid import incompressible_flow

''' Systems ''' 

def tri_poiseuille():
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

def sq_poiseuille():
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

def hex_poiseuille():
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


''' Testing functions ''' 

def render():
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
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True)

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
	render()