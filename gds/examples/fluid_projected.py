'''
Incompressible hydrodynamics by Leray projection method.
'''

import networkx as nx
import numpy as np
import pdb
from itertools import count
import colorcet as cc
import random

from gds.types import *
from .fluid import fluid_test
import gds

''' Definitions ''' 

def navier_stokes(G: nx.Graph, viscosity=1e-3, density=1.0, v_free=[], body_force=None, advect=None, **kwargs) -> (gds.node_gds, gds.edge_gds):
	if body_force is None:
		body_force = lambda t, y: 0
	if advect is None:
		advect = lambda v: v.advect()

	pressure = gds.node_gds(G, **kwargs)
	pressure.set_evolution(lhs=lambda t, p: pressure.laplacian(p) / density, refresh_cvx=False)

	v_free = np.array([pressure.X[x] for x in set(v_free)], dtype=np.intp)
	velocity = gds.edge_gds(G, v_free=v_free, **kwargs)
	velocity.set_evolution(
		dydt=lambda t, u: 
			velocity.leray_project(
				-advect(velocity) + body_force(t, u) + velocity.laplacian() * viscosity/density
			) - pressure.grad() / density,
		max_step=np.inf
	)

	return velocity, pressure

def stokes(G: nx.Graph, **kwargs) -> (gds.node_gds, gds.edge_gds):
	return navier_stokes(G, advect=lambda v: 0., **kwargs)

def euler(G: nx.Graph, **kwargs) -> (gds.node_gds, gds.edge_gds):
	return navier_stokes(G, viscosity=0, **kwargs)


''' Systems ''' 

def test1():
	v_field = {
		(1,2): 2, (2,3): 1, (3,4): 2, (1,4): -3,
		(5,6): 2, (6,7): 3, (7,8): 2, (5,8): -1,
		(1,5): 1, (2,6): 1, (3,7): -1, (4,8): -1,
	}
	G = gds.flat_cube()
	velocity = navier_stokes(G, viscosity=0.)
	velocity.set_initial(y0=lambda e: v_field[e])
	return velocity

def poiseuille():
	m=14 
	n=28 
	gradP=6.0
	viscosity=100

	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	v_free_l = set(l.nodes()) - (set(t.nodes()) | set(b.nodes()))
	v_free_r = set(r.nodes()) - (set(t.nodes()) | set(b.nodes()))
	v_free = v_free_l | v_free_r

	velocity, pressure = navier_stokes(G, viscosity=viscosity, v_free=v_free)

	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.zero_edge_bc(t),
		gds.zero_edge_bc(b),
	))
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes},
		{n: -gradP/2 for n in r.nodes},
	))
	return velocity, pressure

def lid_driven_cavity():
	m=18
	n=21
	v=10.0
	G, (l, r, t, b) = gds.triangular_lattice(m, n*2, with_boundaries=True)
	# t.remove_nodes_from([(0, m), (1, m), (n-1, m), (n, m)])
	velocity, pressure = navier_stokes(G, viscosity=200.)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		gds.const_edge_bc(t, v),
		gds.zero_edge_bc(b),
		gds.zero_edge_bc(l),
		gds.zero_edge_bc(r),
	))
	return velocity, pressure

def euler_test_1():
	v_field = {
		(1,2): 2, (2,3): 1, (3,4): 2, (1,4): -3,
		(5,6): 2, (6,7): 3, (7,8): 2, (5,8): -1,
		(1,5): 1, (2,6): 1, (3,7): -1, (4,8): -1,
	}
	G = gds.flat_cube()
	velocity, pressure = euler(G)
	velocity.set_initial(y0=lambda e: v_field[e])
	return velocity, pressure

def random_euler(G, KE=1.):
	assert KE >= 0
	advector = lambda v: v.advect()
	velocity, pressure = euler(G, advect=advector)
	velocity.advector = advector # TODO: hacky
	y0 = np.random.uniform(low=1, high=2, size=len(velocity))
	y0 = velocity.leray_project(y0)
	y0 *= np.sqrt(KE / np.dot(y0, y0))
	velocity.set_initial(y0=lambda e: y0[velocity.X[e]])

	return velocity, pressure


if __name__ == '__main__':
	gds.set_seed(1)
	# G = gds.torus()
	# G = gds.flat_prism(k=4)
	# G = gds.flat_prism(k=6, n=8)
	# G = gds.icosphere()
	# G = nx.Graph()
	# G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(0,4),(4,5),(5,3)])
	G = gds.triangular_lattice(m=1, n=2)
	# G = nx.random_geometric_graph(40, 0.5)
	# G = gds.voronoi_lattice(10, 100, eps=0.07)

	# G.faces = [tuple(f) for f in nx.cycle_basis(G)]

	# fluid_test(*lid_driven_cavity())
	fluid_test(*random_euler(G, 10))
