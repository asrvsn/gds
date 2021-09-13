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

def navier_stokes(G: nx.Graph, viscosity=1e-3, density=1.0, v_free=[], body_force=None, **kwargs) -> (gds.node_gds, gds.edge_gds):
	if body_force is None:
		body_force = lambda t, y: 0

	pressure = gds.node_gds(G, **kwargs)
	pressure.set_evolution(lhs=lambda t, p: pressure.laplacian(p) / density)

	v_free = np.array([pressure.X[x] for x in set(v_free)], dtype=np.intp)
	velocity = gds.edge_gds(G, **kwargs)
	velocity.set_evolution(dydt=lambda t, u: 
		velocity.leray_project(
			-velocity.advect() + velocity.laplacian() * viscosity/density + body_force(t, u)
		) - pressure.grad() / density
	)

	return velocity, pressure

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

if __name__ == '__main__':
	fluid_test(*poiseuille())

	# m=14 
	# n=28 
	# gradP=6.0
	# G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	# pressure = gds.node_gds(G)
	# pressure.set_evolution(lhs=lambda t, p: pressure.laplacian(p))
	# pressure.set_constraints(dirichlet=gds.combine_bcs(
	# 	{n: gradP/2 for n in l.nodes},
	# 	{n: -gradP/2 for n in r.nodes},
	# ))
	# gds.render(pressure, dynamic_ranges=True)
