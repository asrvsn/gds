import networkx as nx
import numpy as np
import pdb
from itertools import count
import colorcet as cc
import random

from gpde import *
from gpde.utils import set_seed
from gpde.utils.graph import *
from gpde.render.bokeh import *

''' Definitions ''' 

def incompressible_flow(G: nx.Graph, dG: nx.Graph, viscosity=1.0e-3, density=1.0) -> (vertex_pde, edge_pde):
	''' 
	G: graph
	dG: non-divergence-free boundary (inlets/outlets)
	''' 
	velocity = edge_pde(G, dydt=lambda t, self: None, atol=1e-4) # Increased error tolerance for velocity of diff. scales
	dG_nodes = np.array([velocity.nodes[n] for n in dG.nodes], dtype=np.intp)

	def pressure_fun(t, self):
		# TODO: check correctness
		div1 = -self.incidence@(velocity.y/velocity.dt - velocity.advect())
		div2 = velocity.div()
		div1[dG_nodes], div2[dG_nodes] = 0., 0. # do not enforce divergence-free constraint at specified boundaries
		return div1 - self.laplacian()/density + viscosity*self.vertex_laplacian@div2/density
	pressure = vertex_pde(G, lhs=pressure_fun, gtol=1e-8)

	def velocity_fun(t, self):
		return -self.advect() - pressure.grad()/density + viscosity*self.laplacian()/density
	velocity.dydt_fun = velocity_fun

	return pressure, velocity

def compressible_flow(G: nx.Graph, viscosity=1.0) -> (vertex_pde, edge_pde):
	pass

def lagrangian_tracer(velocity: edge_pde, inlets: list) -> vertex_pde:
	''' Passive tracer ''' 
	tracer = vertex_pde(velocity.G, dydt=lambda t, self: -self.advect(velocity))
	tracer.set_boundary(dirichlet=dict_fun({i: 1.0 for i in inlets}))
	return tracer

def const_velocity(dG: nx.Graph, v: float) -> Callable:
	''' Create constant-velocity condition graph boundary ''' 
	def vel(e):
		if e in dG.edges or (e[1], e[0]) in dG.edges: 
			return v
		return None
	return vel

def no_slip(dG: nx.Graph) -> Callable:
	''' Create no-slip velocity condition graph boundary ''' 
	return const_velocity(dG, 0.)

def multi_bc(bcs: List[Callable]) -> Callable:
	def fun(x):
		for bc in bcs:
			v = bc(x)
			if v is not None: return v
		return None
	return fun

''' Systems ''' 

def fluid_on_grid():
	G = grid_graph(10, 10)
	i, o = (3,3), (6,6)
	dG = nx.Graph()
	dG.add_nodes_from([i, o])
	pressure, velocity = incompressible_flow(G, dG)
	pressure.set_boundary(dirichlet=dict_fun({i: 10.0, o: -10.0}))
	tracer = lagrangian_tracer(velocity, [i])
	return pressure, velocity, tracer

def fluid_on_circle():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	pressure, velocity = incompressible_flow(G)
	def pressure_values(x):
		if x == 0:
			return 1.0
		if x == n-1:
			return -1.0
		return None
	pressure.set_boundary(dirichlet=pressure_values)
	return pressure, velocity

def differential_inlets():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4,5,6])
	G.add_edges_from([(1,2),(2,3),(4,3),(2,5),(3,5),(5,6)])
	dG = nx.Graph()
	dG.add_nodes_from([1, 4, 6])
	pressure, velocity = incompressible_flow(G, dG)
	p_vals = {1: 1.0, 4: 0.5, 6: -1.0}
	pressure.set_boundary(dirichlet=dict_fun(p_vals))
	return pressure, velocity

def poiseuille(m=10, n=23, gradP: float=1.0):
	''' Pressure-driven flow by gradient across dG_L -> dG_R ''' 
	assert n % 2 == 1
	G = lattice45(m, n)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	dG_L.remove_nodes_from([(1, j) for j in range(m)])
	dG_R.remove_nodes_from([(n-2, j) for j in range(m)])
	pressure, velocity = incompressible_flow(G, nx.compose_all([dG_L, dG_R]))
	def pressure_values(x):
		if x in dG_L.nodes:
			return gradP/2
		elif x in dG_R.nodes:
			return -gradP/2
		return None
	pressure.set_boundary(dirichlet=pressure_values)
	velocity.set_boundary(dirichlet=no_slip(nx.compose_all([dG_T, dG_B])))
	return pressure, velocity

def poiseuille_asymmetric(m=12, n=24, gradP: float=1.0):
	''' Poiseuille flow with a boundary asymmetry '''
	G = grid_graph(m, n)
	k = 6
	blockage = [
		(k, m-1), (k, m-2),
		(k, m-2), (k, m-3),
		(k, m-3), (k+1, m-3),
		(k+1, m-3), (k+1, m-2),
		(k+1, m-2), (k+1, m-1),
	]
	nbs = nx.node_boundary(G, blockage) | {(k-1, m-4), (k+2, m-4)}
	G.remove_nodes_from(blockage)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	dG_T.add_nodes_from(nbs)
	dG_T.add_edges_from(nx.edge_boundary(G, nbs, nbs))
	pressure, velocity = incompressible_flow(G, nx.compose_all([dG_L, dG_R]))
	def pressure_values(x):
		if x[0] == 0: return gradP/2
		if x[0] == n-1: return -gradP/2
		return None
	pressure.set_boundary(dirichlet=pressure_values)
	velocity.set_boundary(dirichlet=no_slip(nx.compose_all([dG_T, dG_B])))
	return pressure, velocity

def couette(G: nx.Graph, dG_l: nx.Graph, dG_w: nx.Graph, v_l=1.0):
	''' Drag-induced flow by velocity on dG_l with no-slip on dG_w ''' 
	pressure, velocity = incompressible_flow(G)
	wall = noslip(dG_w)
	lid_edges = set(dG_l.edges())
	def bc(t, e):
		v = wall(e)
		if v is None:
			if e in lid_edges:
				return v_l
			elif (e[1], e[0]) in lid_edges:
				return -v_l
		return None
	velocity.set_boundary(dirichlet=bc)
	return pressure, velocity

def von_karman(m=12, n=30, gradP=10.0):
	G = grid_graph(m, n)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	j, k = 6, int(m/2)
	obstacle = [ # Introduce occlusion
		(j, k), 
		(j+1, k-1), (j+1, k), 
		(j+2, k-1),
	]
	G.remove_nodes_from(obstacle)
	pressure, velocity = incompressible_flow(G, nx.compose_all([dG_L, dG_R]))
	def pressure_values(x):
		if x[0] == 0: return gradP/2
		if x[0] == n-1: return -gradP/2
		return None
	pressure.set_boundary(dirichlet=pressure_values)
	velocity.set_boundary(dirichlet=no_slip(nx.compose_all([dG_L, dG_T, dG_B])))
	tracer = lagrangian_tracer(velocity, [(0, 2*i+1) for i in range(int(m/2))])
	return pressure, velocity, tracer

def random_graph():
	set_seed(1001)
	n = 30
	eps = 0.3
	G = nx.random_geometric_graph(n, eps)
	pressure, velocity = incompressible_flow(G)
	pressure.set_boundary(dirichlet=dict_fun({4: 1.0, 21: -1.0}))
	return pressure, velocity

def test1():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4])
	G.add_edges_from([(1,2),(2,3),(3,4)])
	pressure, velocity = incompressible_flow(G)
	p_vals = {}
	v_vals = {(1, 2): 1.0, (2, 3): 1.0}
	pressure.set_boundary(dirichlet=dict_fun(p_vals))
	velocity.set_boundary(dirichlet=dict_fun(v_vals))
	return pressure, velocity

def test2():
	n = 20
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	pressure, velocity = incompressible_flow(G, nx.Graph(), viscosity=0.)
	velocity.set_initial(y0=dict_fun({(2,3): 1.0, (3,4): 1.0}, def_val=0.))
	return pressure, velocity

def lid_driven_cavity(m=15, n=25, v=1.0):
	G = grid_graph(m, n)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	cavity = nx.compose_all([dG_L, dG_R, dG_B])
	lid = dG_T
	pressure, velocity = incompressible_flow(G, nx.Graph())
	velocity.set_boundary(dirichlet=multi_bc([no_slip(cavity), const_velocity(lid, v)]))
	return pressure, velocity

''' Experimentation / observation ''' 

class TurbulenceObservable(Observable):
	def __init__(self, velocity: edge_pde):
		self.velocity = velocity
		self.metrics = {
			'n': 0,
			'v_mu': 0.,
			'v_M2': 0.,
			'v_sigma': 0,
		}
		# Warning: do not allow rendered metrics to be None, or Bokeh won't render it
		self.rendered = ['v_sigma',]
		self.cycle_indices = {}
		self.cycle_signs = {}
		for cycle in nx.cycle_basis(velocity.G):
			n = len(cycle)
			id = f'{n}-cycle flow'
			self.metrics[id] = 0.
			G_cyc = nx.Graph()
			nx.add_cycle(G_cyc, cycle)
			cyc_orient = {}
			for i in range(n):
				if i == n-1:
					cyc_orient[(cycle[i], cycle[0])] = 1
					cyc_orient[(cycle[0], cycle[i])] = -1
				else:
					cyc_orient[(cycle[i], cycle[i+1])] = 1
					cyc_orient[(cycle[i+1], cycle[i])] = -1
			indices = [velocity.edges[e] for e in G_cyc.edges()]
			signs = np.array([velocity.orientation[e]*cyc_orient[e] for e in G_cyc.edges()])
			if id in self.cycle_indices:
				self.cycle_indices[id].append(indices)
				self.cycle_signs[id].append(signs)
			else:
				self.cycle_indices[id] = [indices]
				self.cycle_signs[id] = [signs]
		self.rendered += list(self.cycle_indices.keys())
		super().__init__({})

	def observe(self):
		y = self.velocity.y
		if self['n'] == 0:
			self['v_mu'] = y
			self['v_M2'] = np.zeros_like(y)
			self['v_sigma'] = 0.
		else:
			n = self['n'] + 1
			new_mu = self['v_mu'] + (y - self['v_mu']) / n
			self['v_M2'] += (y - self['v_mu']) * (y - new_mu)
			self['v_sigma'] = np.sqrt((self['v_M2'] / n).sum())
			self['v_mu'] = new_mu
		self['n'] += 1
		for id, cycles in self.cycle_indices.items():
			self[id] = sum([(y[cyc] ** 2).sum() for cyc in cycles])
		return self.y

	def __getitem__(self, idx):
		return self.metrics.__getitem__(idx)

	def __setitem__(self, idx, val):
		return self.metrics.__setitem__(idx, val)

	@property 
	def t(self):
		return self.velocity.t

	@property
	def y(self):
		ret = {k: [self.metrics[k]] for k in self.rendered}
		ret['t'] = [self.t]
		return ret
	
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
import colorcet as cc

class FluidRenderer(Renderer):
	def __init__(self, pressure: vertex_pde, velocity: edge_pde, **kwargs):
		self.pressure = pressure
		self.velocity = velocity
		self.turbulence = TurbulenceObservable(velocity)
		sys = couple(pressure, velocity)
		super().__init__(sys, **kwargs)

	def setup_canvas(self):
		return [
			[[[self.pressure, self.velocity]], [[self.turbulence]]]
		]

	def create_plot(self, items: List[Observable]):
		if len(items) == 1 and items[0] is self.turbulence:
			cats = list(self.turbulence.rendered)
			src = ColumnDataSource({cat: [] for cat in ['t'] + cats})
			plots = []
			for i, cat in enumerate(cats):
				plot = figure(title=cat, tooltips=[(cat, '@'+cat)])
				if i == 0:
					plot.toolbar_location = 'above'
					plot.x_range.follow = 'end'
					plot.x_range.follow_interval = 10.0
					plot.x_range.range_padding = 0
				else:
					plot.toolbar_location = None
					plot.x_range = plots[0].x_range
				plot.line('t', cat, line_color='black', source=src)
				# plot.varea(x='t', y1=0, y2=cat, fill_color=cc.glasbey[i], alpha=0.6, source=src)
				plots.append(plot)
			self.turbulence.src = src
			return column(plots, sizing_mode='stretch_both')
		else:
			return super().create_plot(items)

	def draw(self):
		self.turbulence.src.stream(self.turbulence.observe(), 200)
		super().draw()

if __name__ == '__main__':
	''' Solve ''' 

	p, v = poiseuille(gradP=5.0)
	# p, v = poiseuille_asymmetric(gradP=10.0)
	# p, v = lid_driven_cavity(v=10.)
	# p, v, t = fluid_on_grid()
	# p, v = differential_inlets()
	# p, v = von_karman(n=50, gradP=20)
	# p, v = test2()

	d = v.project(GraphDomain.nodes, lambda v: v.div()) # divergence of velocity
	a = v.project(GraphDomain.edges, lambda v: v.advect()) # advective strength
	f = v.project(GraphDomain.nodes, lambda v: v.influx()) # mass flux through nodes; assumes divergence-free flow
	# g = p.project(GraphDomain.edges, lambda p: p.grad())
	pv = couple(p, v)
	sys = System(pv, {
		'pressure': p,
		'velocity': v,
		# 'divergence': d,
		'mass flux': f,
		'advection': a,
		# 'tracer': t,
		# 'grad': grad,
	})

	''' Save to disk ''' 
	# sys.solve_to_disk(20, 1e-2, 'poiseuille')

	''' Load from disk ''' 
	# sys = System.from_disk('von_karman')
	# p, v, d, a = sys.observables['pressure'], sys.observables['velocity'], sys.observables['divergence'], sys.observables['advection']

	renderer = LiveRenderer(sys, [[[[p, v]], [[f]], [[a]]]], node_palette=cc.rainbow, node_rng=(-1,1), node_size=0.03)
	renderer.start()
