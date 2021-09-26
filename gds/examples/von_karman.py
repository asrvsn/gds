'''
Von Karman (vortex shedding) flow
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
import gds.examples.fluid as fluid
import gds.examples.fluid_projected as fluid_projected

''' Systems ''' 

def von_karman():
	m=24 
	n=113 
	gradP=10.0
	inlet_v = 5.0
	outlet_p = 0.0
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
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
	velocity, pressure = fluid.navier_stokes(G, viscosity=1e-4, v_free=(l.nodes | r.nodes))
	pressure.set_constraints(dirichlet=gds.combine_bcs(
		# {n: gradP/2 for n in l.nodes},
		# {n: -gradP/2 for n in r.nodes}
		{n: 0 for n in r.nodes}
		# {(n//2+1, j): outlet_p for j in range(n)}
	))
	gradation = np.linspace(-0.5, 0.5, m+1)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		{((0, i), (1, i)): inlet_v + gradation[i] for i in range(1, m)},
		# {((n//2, i), (n//2+1, i)): inlet_v - gradation[i] for i in range(1, m)},
		# {((n//2-1, 2*i+1), (n//2, 2*i+1)): inlet_v - gradation[2*i+1] for i in range(0, m//2)},
		gds.utils.bidict({e: 0 for e in obstacle_boundary}),
		gds.utils.bidict({e: 0 for e in t.edges}),
		gds.utils.bidict({e: 0 for e in b.edges})

	))
	return velocity, pressure

def von_karman_projected():
	m=24 
	n=113 
	gradP=100.0
	inlet_v = 5.0
	outlet_p = 0.0

	# G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	G = gds.triangular_cylinder(m, n)
	l, r = G.l_boundary, G.r_boundary

	j, k = 8, m//2
	# Introduce occlusion
	obstacle = [ 
		(j, k), 
		(j+1, k),
		(j, k+1), 
		(j, k-1),
		(j-1, k+1), 
		# (j+1, k+1), 
		# (j+1, k-1),
		(j, k+2), 
		# (j, k-2), 
	]
	obstacle_boundary = gds.utils.flatten([G.neighbors(n) for n in obstacle])
	obstacle_boundary = list(nx.edge_boundary(G, obstacle_boundary, obstacle_boundary))
	G.remove_nodes_from(obstacle)
	velocity, pressure = fluid_projected.navier_stokes(G, viscosity=0.1, density=1, v_free=(l.nodes | r.nodes))

	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes},
		{n: -gradP/2 for n in r.nodes}
	))

	# gradation = np.linspace(-0.5, 0.5, m+1)
	# velocity.set_constraints(dirichlet=gds.combine_bcs(
		# {((0, i), (1, i)): inlet_v + gradation[i] for i in range(1, m)},
		# {((n//2, i), (n//2+1, i)): inlet_v - gradation[i] for i in range(1, m)},
		# {((n//2-1, 2*i+1), (n//2, 2*i+1)): inlet_v - gradation[2*i+1] for i in range(0, m//2)},
	# 	gds.utils.bidict({e: 0 for e in obstacle_boundary}),
	# 	gds.utils.bidict({e: 0 for e in t.edges}),
	# 	gds.utils.bidict({e: 0 for e in b.edges})
	# ))
	return velocity, pressure


''' Testing functions ''' 

def render():
	pass

def dump():
	pass

if __name__ == '__main__':
	# fluid.fluid_test(*von_karman())
	fluid.fluid_test(*von_karman_projected())