'''
Incompressible hydrodynamics by Leray projection method.
'''

import networkx as nx
import numpy as np
import pdb
from itertools import count, chain
import colorcet as cc
import random
from scipy import stats

from gds.types import *
from gds.utils import relu, rotate
from .fluid import fluid_test, initial_flow
import gds

''' Definitions ''' 

def navier_stokes(G: nx.Graph, viscosity=1e-3, density=1.0, v_free=[], body_force=None, advect=None, integrator=Integrators.lsoda, **kwargs) -> (gds.node_gds, gds.edge_gds):
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
				-advect(velocity) + body_force(t, u) + (0 if viscosity == 0 else velocity.laplacian() * viscosity/density)
			) - pressure.grad() / density,
		max_step=1e-3,
		integrator=integrator,
	)

	return velocity, pressure

def stokes(G: nx.Graph, **kwargs) -> (gds.node_gds, gds.edge_gds):
	return navier_stokes(G, advect=lambda v: 0., **kwargs)

def euler(G: nx.Graph, **kwargs) -> (gds.node_gds, gds.edge_gds):
	# Use L2 norm-conserving symmetric integrator
	return navier_stokes(G, viscosity=0, integrator=Integrators.implicit_midpoint, **kwargs)


def euler_cycles(G: nx.Graph, density=1.0, integrator=Integrators.lsoda):
	velocity = gds.edge_gds(G)
	cycles = gds.cycle_basis(G)
	print(cycles)
	ops = dict()
	edge_set = set(G.edges())
	for i, cycle in enumerate(cycles):
		print(cycle)
		n = len(cycle)
		D = np.zeros((n, n))
		P = np.zeros((n, velocity.ndim))
		# edge_pairs = zip(zip(chain([cycle[-2], cycle[-1]], cycle[:-2]), chain([cycle[-1]], cycle[:-1])), zip(chain([cycle[-1]], cycle[:-1]), cycle))
		edges = zip(chain([cycle[-1]], cycle[:-1]), cycle)
		for idx, e_i in enumerate(edges):
			D[idx,idx] = -1
			D[idx,(idx+1)%n] = 1
			i = velocity.X[e_i]
			P[idx,i] = 1 if e_i in edge_set else -1 # Re-orient
		Dm = relu(-D)
		Dp = relu(D)
		F = Dm.T @ Dp - Dp.T @ Dm
		ops[i] = {
			'c': cycle,
			'D': D,
			'P': P,
			'F': F
		}
	def dvdt(t, v):
		ret = 0
		for i in ops:
			F, P = ops[i]['F'], ops[i]['P']
			pv = P @ v
			ret -= P.T @ (np.multiply(F.T, pv).T + np.multiply(F, pv)) @ pv
		return velocity.leray_project(ret)
	# dvdt(0, velocity.y)
	velocity.set_evolution(dydt=dvdt, integrator=integrator)
	return velocity


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

def random_euler(G, **kwargs):
	velocity, pressure = euler(G)
	y0 = initial_flow(G, **kwargs)
	velocity.set_initial(y0=lambda e: y0[velocity.X[e]])
	return velocity, pressure

def ns_cycle_test():
	n = 20
	G = gds.directed_cycle_graph(n)
	velocity = gds.edge_gds(G)
	print(velocity.X.keys())
	D = np.zeros((velocity.ndim, velocity.ndim))
	IV = np.zeros((velocity.ndim, velocity.ndim))
	edge_pairs = zip(zip(chain([n-2, n-1], range(n-2)), chain([n-1], range(n-1))), zip(chain([n-1], range(n-1)), range(n)))
	for idx, (e_i, e_j) in enumerate(edge_pairs):
		print(e_i, e_j)
		i, j = velocity.X[e_i], velocity.X[e_j]
		D[i,i] = -1
		D[i,j] = 1
		IV[j,idx] = 1
	# print(D)
	# D = -velocity.incidence # Either incidence or dual derivative seems to work
	Dm = relu(-D)
	Dp = relu(D)
	F = Dm.T @ Dp - Dp.T @ Dm

	# pdb.set_trace()
	def dvdt(t, v):
		A = np.multiply(F.T, v).T + np.multiply(F, v)
		return -A @ v

	velocity.set_evolution(dydt=dvdt)
	bump = stats.norm().pdf(np.linspace(-4, 4, n)) 
	v0 = -IV @ bump
	# v0 = IV @ rotate(bump, 10)
	# v0 = IV @ (bump - rotate(bump, 10))
	velocity.set_initial(y0=lambda e: v0[velocity.X[e]])

	sys = gds.couple({
		'velocity': velocity,
		'gradient': velocity.project(gds.GraphDomain.edges, lambda v: D @ v.y),
		# 'laplacian': velocity.project(gds.GraphDomain.edges, lambda v: -D.T @ D @ v.y),
		'dual': velocity.project(gds.GraphDomain.nodes, lambda v: v.y),
		'L1': velocity.project(PointObservable, lambda v: np.abs(v.y).sum()),
		'L2': velocity.project(PointObservable, lambda v: np.linalg.norm(v.y)),
		'min': velocity.project(PointObservable, lambda v: v.y.min()),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=3)


if __name__ == '__main__':
	gds.set_seed(1)
	# G = gds.torus()
	G = gds.flat_prism(k=4)
	# G = gds.flat_prism(k=6, n=8)
	# G = gds.icosphere()
	# G = nx.Graph()
	# G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(0,4),(4,5),(5,3)])
	# G = gds.triangular_lattice(m=1, n=4)
	# G = nx.random_geometric_graph(40, 0.5)
	# G = gds.voronoi_lattice(10, 100, eps=0.07)

	# G.faces = [tuple(f) for f in nx.cycle_basis(G)]

	# v, p = random_euler(G, KE=10)
	# T = v.advection_tensor()
	# pdb.set_trace()

	# fluid_test(v, p)

	# ns_cycle_test()

	y0 = initial_flow(G, KE=10)
	velocity = euler_cycles(G)
	velocity.set_initial(y0=lambda e: y0[velocity.X[e]])
	fluid_test(velocity)
