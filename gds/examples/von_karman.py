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
	inlet_v = 10.0
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
	gradation = np.linspace(-0.1, 0.1, m+1)
	velocity.set_constraints(dirichlet=gds.combine_bcs(
		{((0, i), (1, i)): inlet_v + gradation[i] for i in range(1, m)},
		# {((n//2, i), (n//2+1, i)): inlet_v - gradation[i] for i in range(1, m)},
		# {((n//2-1, 2*i+1), (n//2, 2*i+1)): inlet_v - gradation[2*i+1] for i in range(0, m//2)},
		# gds.utils.bidict({e: 0 for e in obstacle_boundary}),
		gds.utils.bidict({e: 0 for e in t.edges}),
		gds.utils.bidict({e: 0 for e in b.edges})

	))

	sys = gds.couple({
		'velocity': velocity,
		# 'divergence': velocity.project(gds.GraphDomain.nodes, lambda v: v.div()),
		# 'vorticity': velocity.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'pressure': pressure,
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True, edge_colors=True, edge_palette=cc.bgy)


def von_karman_projected():
	m=20
	n=40 
	gradP=100.0
	inlet_v = 5.0
	outlet_p = 0.0

	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	# G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	# G, (l, r, t, b) = gds.hexagonal_lattice(m, n, with_boundaries=True)
	# G = gds.triangular_cylinder(m, n)
	# G = gds.square_cylinder(m, n)
	# l, r = G.l_boundary, G.r_boundary

	j, k = 3, m//2
	# Introduce occlusion
	obstacle = [ 
		(j, k), 
		# (j+1, k),
		# (j+1, k+1), 
		# (j+1, k-1),
		# (j-1, k+1), 
		# (j, k+2), 
	]
	G.remove_nodes_from(obstacle)
	velocity, pressure = fluid_projected.navier_stokes(G, viscosity=0.0001)

	pressure.set_constraints(dirichlet=gds.combine_bcs(
		{n: gradP/2 for n in l.nodes},
		{n: -gradP/2 for n in r.nodes}
	))

	# velocity.set_constraints(dirichlet=gds.combine_bcs(
	# 	{((0, i), (1, i)): inlet_v for i in range(1, m)},
	# 	{((n//2, i), (n//2+1, i)): inlet_v for i in range(1, m)},
	# 	{((n//2-1, 2*i+1), (n//2, 2*i+1)): inlet_v for i in range(0, m//2)},
	# 	# gds.utils.bidict({e: 0 for e in obstacle_boundary}),
	# 	gds.utils.bidict({e: 0 for e in t.edges}),
	# 	gds.utils.bidict({e: 0 for e in b.edges})
	# ))

	sys = gds.couple({
		'velocity': velocity,
		# 'divergence': velocity.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'vorticity': velocity.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'pressure': pressure,
		# 'tracer': lagrangian_tracer(velocity),
		# 'advective': velocity.project(gds.GraphDomain.edges, lambda v: -advector(v)),
		# 'L2': velocity.project(PointObservable, lambda v: np.sqrt(np.dot(v.y, v.y))),
		# 'dK/dt': velocity.project(PointObservable, lambda v: np.dot(v.y, v.leray_project(-advector(v)))),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True, edge_colors=True, edge_palette=cc.bgy)


def euler_vortex_street():
	'''
	Vortex street translation in the inviscid model.
	x-periodic hexagonal lattice.
	'''
	# m, n = 4, 20
	# G = gds.hexagonal_lattice(m, n)
	# speed = 1
	# # G = gds.contract_pairs(G, [((0, j), (n, j)) for j in range(1, 2*m)])
	# # G = gds.remove_pos(G)

	# velocity, pressure = fluid_projected.euler(G)
	# y0 = np.zeros(velocity.ndim)
	# for j in range(n-1):
	# 	if j == 5:
	# 		e = ((j, m), (j, m+1))
	# 		y_e = np.zeros(velocity.ndim)
	# 		y_e[velocity.X[e]] = 1
	# 		y_f = speed * velocity.curl_face@y_e
	# 		y_e = velocity.curl_face.T@y_f
	# 		y0 += y_e

	m, n = 9, 20
	G = gds.square_lattice(m, n)
	speed = 1

	velocity, pressure = fluid_projected.euler(G)
	y0 = np.zeros(velocity.ndim)

	for j in [m//2, m//2+1]:
		i = 5
		e = ((i, j), (i+1, j))
		y_e = np.zeros(velocity.ndim)
		y_e[velocity.X[e]] = 1
		y_f = speed * velocity.curl_face@y_e
		y0 += velocity.curl_face.T@y_f

	velocity.set_initial(y0=lambda x: y0[velocity.X[x]])

	sys = gds.couple({
		'velocity': velocity,
		'vorticity': velocity.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'advective': velocity.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		'pressure': pressure,
		# 'divergence': velocity.project(gds.GraphDomain.nodes, lambda v: v.div()),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True, edge_colors=True, edge_palette=cc.bgy, n_spring_iters=2000)


''' Testing functions ''' 

def render():
	pass

def dump():
	pass

if __name__ == '__main__':
	# von_karman()
	# von_karman_projected()
	euler_vortex_street()