import networkx as nx
import numpy as np
import pdb
from itertools import count
import colorcet as cc
import random
import scipy.sparse as sp
from scipy.optimize import minimize

from gds.types import *
import gds

''' Definitions ''' 

def navier_stokes(G: nx.Graph, viscosity=1e-3, density=1.0, v_free=[], e_free=[], e_normal=[], advect=None, integrator=Integrators.lsoda, **kwargs) -> (gds.node_gds, gds.edge_gds):
	if advect is None:
		advect = lambda v: v.advect()

	pressure = gds.node_gds(G, **kwargs)
	velocity = gds.edge_gds(G, **kwargs)
	v_free = np.array([pressure.X[x] for x in set(v_free)], dtype=np.intp)		# Inlet/outlet nodes (typically, pressure boundaries)
	e_free = np.array([velocity.X[x] for x in set(e_free)], dtype=np.intp) 		# Free-slip surface
	e_normal = np.array([velocity.X[x] for x in set(e_normal)], dtype=np.intp) 	# Inlets/outlet edges normal to surface
	min_step = 1e-3

	def pressure_f(t, y):
		dt = max(min_step, velocity.dt)
		lhs = velocity.div(velocity.y/dt - advect(velocity) + velocity.laplacian(free=e_free, normal=e_normal) * viscosity/density)
		lhs[v_free] = 0.
		lhs -= pressure.laplacian(y)/density 
		return lhs

	def velocity_f(t, y):
		return -advect(velocity) - pressure.grad()/density + velocity.laplacian(free=e_free, normal=e_normal) * viscosity/density

	pressure.set_evolution(lhs=pressure_f)
	velocity.set_evolution(dydt=velocity_f, integrator=integrator)

	return velocity, pressure

def stokes(G: nx.Graph, **kwargs) -> (gds.node_gds, gds.edge_gds):
	return navier_stokes(G, advect=lambda v: 0., **kwargs)

def euler(G: nx.Graph, **kwargs) -> (gds.node_gds, gds.edge_gds):
	# Use L2 norm-conserving symmetric integrator
	return navier_stokes(G, viscosity=0, integrator=Integrators.implicit_midpoint, **kwargs)

def lagrangian_tracer(velocity: gds.edge_gds, alpha=1.0) -> gds.node_gds:
	''' Passive tracer ''' 
	tracer = gds.node_gds(velocity.G)
	tracer.set_evolution(dydt=lambda t, y: -alpha*tracer.advect(velocity))
	start_x = random.choice(list(velocity.G.nodes()))
	tracer.set_initial(y0=lambda x: 1. if x == start_x else 0.)
	return tracer

''' Systems ''' 

def fluid_on_grid():
	G = gds.square_lattice(10, 10)
	i, o = (3,3), (6,6)
	dG = nx.Graph()
	dG.add_nodes_from([i, o])
	velocity, pressure = navier_stokes(G, dG)
	pressure.set_constraints(dirichlet={i: 10.0, o: -10.0})
	tracer = lagrangian_tracer(velocity, [i])
	return velocity, pressure, tracer

def fluid_on_circle():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	velocity, pressure = navier_stokes(G)
	pressure.set_constraints(dirichlet={0: 1.0, n-1: -1.0})
	return velocity, pressure

def differential_inlets():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4,5,6])
	G.add_edges_from([(1,2),(2,3),(4,3),(2,5),(3,5),(5,6)])
	velocity, pressure = navier_stokes(G, inlets=[1,4], outlets=[6], viscosity=1e-2,)
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
	return velocity, pressure

def differential_outlets():
	G = nx.Graph()
	G.add_nodes_from(list(range(8)))
	G.add_edges_from([
		(0, 1), 
		(1, 2), (2, 4), (4, 6),
		(1, 3), (3, 5), (5, 7),
		(4, 5)
	])
	velocity, pressure = navier_stokes(G, inlets=[0], outlets=[6, 7], viscosity=0.01)
	def positive_outlets(vel):
		for outlet in [(4, 6), (5, 7)]:
			vel[velocity.X[outlet]] = max(0, vel[velocity.X[outlet]])
		return vel
	velocity.set_constraints(dirichlet={(0, 1): 1.0}, project=positive_outlets)
	pressure.set_constraints(dirichlet={6: 2.0, 7: 1.98})
	return velocity, pressure

def box_inlets():
	G = nx.Graph()
	G.add_nodes_from([0,1,2,3,4,5,6,7])
	G.add_edges_from([(1,2),(2,3),(2,4),(3,5),(4,5),(5,6),(0,3),(4,7)])
	velocity, pressure = navier_stokes(G, inlets=[1,0], outlets=[6,7], viscosity=1.)
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
	return velocity, pressure

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
	velocity, pressure = navier_stokes(G, viscosity=viscosity)
	velocity.set_initial(y0=v_field)
	pressure.set_constraints(dirichlet={0: 0.}) # Pressure reference
	return velocity, pressure


def fluid_test(velocity, pressure=None, columns=3, **kwargs):
	if hasattr(velocity, 'advector'): advector = velocity.advector # TODO: hacky
	else: advector = lambda v: v.advect() 
	freqs, spec_fun = edge_power_spectrum(velocity.G)
	obs = {
		'velocity': velocity,
		'divergence': velocity.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'vorticity': velocity.project(gds.GraphDomain.faces, lambda v: v.curl()),
		# 'diffusion': velocity.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		# 'tracer': lagrangian_tracer(velocity),
		'advective': velocity.project(gds.GraphDomain.edges, lambda v: -advector(v)),
		# 'leray projection': velocity.project(gds.GraphDomain.edges, lambda v: v.leray_project()),
		# 'L1': velocity.project(PointObservable, lambda v: np.abs(v.y).sum()),
		# 'power spectrum': velocity.project(VectorObservable, lambda v: spec_fun(v.y), freqs.tolist()),
		# 'power L2': velocity.project(PointObservable, lambda v: np.sqrt(spec_fun(v.y).sum())),
		# 'L2': velocity.project(PointObservable, lambda v: np.sqrt(np.dot(v.y, v.y))),
		# 'dK/dt': velocity.project(PointObservable, lambda v: np.dot(v.y, -advector(v))),
		# 'dK/dt': velocity.project(PointObservable, lambda v: np.dot(v.y, -advector(velocity) - pressure.grad())),
	}
	if pressure != None:
		obs['pressure'] = pressure
		# obs['pressure_grad'] = pressure.project(gds.GraphDomain.edges, lambda p: -p.grad())
	sys = gds.couple(obs)
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), columns), edge_max=0.6, dynamic_ranges=True, **kwargs)


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
	velocity, pressure = navier_stokes(G, viscosity=100., density=1.0, inlets=l.nodes, outlets=r.nodes, w_key='w')
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
	return velocity, pressure


def random_graph():
	set_seed(1001)
	n = 30
	eps = 0.3
	G = nx.random_geometric_graph(n, eps)
	velocity, pressure = navier_stokes(G)
	pressure.set_constraints(dirichlet=dict_fun({4: 1.0, 21: -1.0}))
	return velocity, pressure

def test1():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4])
	G.add_edges_from([(1,2),(2,3),(3,4)])
	velocity, pressure = navier_stokes(G)
	p_vals = {}
	v_vals = {(1, 2): 1.0, (2, 3): 1.0}
	pressure.set_constraints(dirichlet=dict_fun(p_vals))
	velocity.set_constraints(dirichlet=dict_fun(v_vals))
	return velocity, pressure

def test2():
	n = 20
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	velocity, pressure = navier_stokes(G, viscosity=0.)
	velocity.set_initial(y0=dict_fun({(2,3): 1.0, (3,4): 1.0}, def_val=0.))
	return velocity, pressure

def euler1():
	G = nx.Graph()
	G.add_nodes_from(list(range(1,7)))
	G.add_edges_from([
		(1,2),(2,3),(3,4),(4,1),
		(4,5),(5,6),(6,3),
	])
	negated = set([(1,4),(3,6),])
	def v_field(e):
		ret = 1.0
		if e in negated:
			ret *= -1
		if e == (3,4):
			ret *= 2
		return ret
	velocity, pressure = navier_stokes(G, viscosity=0.)
	velocity.set_initial(y0=v_field)
	return velocity, pressure

def euler2():
	v_field = {
		(1,2): 2, (2,3): 1, (3,4): 2, (1,4): -3,
		(5,6): 2, (6,7): 3, (7,8): 2, (5,8): -1,
		(1,5): 1, (2,6): 1, (3,7): -1, (4,8): -1,
	}
	G = gds.flat_cube()
	velocity, pressure = navier_stokes(G, viscosity=0.)
	velocity.set_initial(y0=lambda e: v_field[e])
	return velocity, pressure

def euler3():
	G = nx.Graph()
	v_field = {
		(1,2): 4, (2,3): 1, (3,4): 4, (4,5): 1, (5,6): 4, (6,7): 1, (7,8): 4, (1,8): -1,
		(9,10): 1, (10,11): 4, (11,12): 1, (12,13): 4, (13,14): 1, (14,15): 4, (15,16): 1, (9, 16): -4,
		(1,9): -3, (2,10): 3, (3,11): -3, (4,12): 3, (5,13): -3, (6,14): 3, (7,15): -3, (8,16): 3, 
	}
	G.add_edges_from(v_field.keys())
	velocity, pressure = navier_stokes(G, viscosity=0.)
	velocity.set_initial(y0=lambda e: v_field[e])
	return velocity, pressure

def random_euler(G, **kwargs):
	velocity, pressure = euler(G)
	y0 = initial_flow(G, **kwargs)
	velocity.set_initial(y0=lambda e: y0[velocity.X[e]])
	return velocity, pressure

def initial_flow(G: nx.Graph, KE: float=1., scale_distribution: Callable=None):
	'''
	Construct divergence-free initial conditions with energies at specified length-scales.

	scale_distribution: probability measure on [0, 1] (default uniform)
	'''
	assert KE >= 0, 'Specify nonnegative kinetic energy'
	if scale_distribution is None:
		scale_distribution = lambda x: 1.

	N = len(G.edges())
	P = gds.edge_gds(G).leray_projector # TODO: assumes determinism of construction
	freqs, spec_fun = edge_power_spectrum(G)
	dist = np.array(list(map(scale_distribution, (freqs - freqs.min()) / (freqs.max() - freqs.min()))))
	def f(x):
		return np.linalg.norm(spec_fun(P @ x) - dist)
	x0 = np.random.uniform(size=N)
	sol = minimize(f, x0)
	u = P @ sol.x
	u *= np.sqrt(KE / np.dot(u, u))
	return u

def edge_fourier_transform(G, method='hodge_faces'):
	'''
	Diagonalization of 1-form laplacian with varying 2-form definitions
	'''
	if method == 'hodge_faces': # 2-forms defined on planar faces (default)
		pass
	elif method == 'hodge_cycles': # 2-forms defined on cycle basis
		G.faces = [tuple(f) for f in nx.cycle_basis(G)]
	elif method == 'dual':
		raise NotImplementedError()
	else:
		raise ValueError(f'Method {method} undefined')

	v = gds.edge_gds(G) # TODO: assumes determinism of edge index assignment -- fix!
	L1 = -v.laplacian(np.eye(v.ndim))
	eigvals, eigvecs = np.linalg.eigh(L1) # Important to use eigh() rather than eig() -- otherwise non-unitary eigenvectors
	eigvecs = np.asarray(eigvecs)

	# Unitary check
	assert (np.round(eigvecs@eigvecs.T, 6) == np.eye(v.ndim)).all(), 'VV^T != I'
	assert (np.round(eigvecs.T@eigvecs, 6) == np.eye(v.ndim)).all(), 'V^TV != I'

	return eigvals, eigvecs


def edge_power_spectrum(G, method='hodge_faces', raw=False):
	'''
	Projection onto eigenspace of Hodge Laplacian.
	'''
	eigvals, eigvecs = edge_fourier_transform(G, method=method)
	evs = sorted(zip(eigvals, eigvecs.T), key=lambda x: np.abs(x[0])) # order by lowest to highest frequency magnitude
	freqs = np.round(np.abs(np.array([x[0] for x in evs])), 4)
	freqs_ = np.unique(freqs)
	A = np.zeros((freqs_.size, freqs.size))
	for i in range(freqs_.size):
		for j in range(freqs.size):
			if freqs_[i] == freqs[j]:
				A[i,j] = 1
	eigspace = np.asarray(np.vstack(tuple(x[1] for x in evs)))
	projector = lambda u: A @ (np.abs(eigspace @ u) ** 2)
	if raw:
		return freqs_, A, eigspace
	else:
		return freqs_, projector


if __name__ == '__main__':
	gds.set_seed(1)
	# G = gds.torus()
	# G = gds.flat_prism(k=4)
	G = gds.flat_prism(k=6, n=6)
	# G = gds.icosphere()
	# G = nx.Graph()
	# G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(0,4),(4,5),(5,3)])

	# fluid_test(*lid_driven_cavity())
	fluid_test(*random_euler_3(G, 10))


