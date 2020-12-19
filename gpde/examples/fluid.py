import networkx as nx
import numpy as np
import pdb
from itertools import count
import colorcet as cc

from gpde import *
from gpde.utils import set_seed
from gpde.utils.graph import *
from gpde.render.bokeh import *

''' Definitions ''' 

def incompressible_flow(G: nx.Graph, viscosity=1.0, density=1.0) -> (vertex_pde, edge_pde):
	velocity = edge_pde(G, dydt=lambda t, self: None)

	def pressure_fun(t, self):
		div = -self.incidence@velocity.advect()
		div[self.dirichlet_indices] = 0. # Don't enforce divergence constraint at boundaries
		return div + self.laplacian()/density
	pressure = vertex_pde(G, lhs=pressure_fun, gtol=1e-8)

	def velocity_fun(t, self):
		# TODO: momentum diffusion here is wrong.
		# return -self.advect() - pressure.grad()/density + viscosity*self.laplacian()/density
		return self.advect() - pressure.grad()/density + viscosity*self.laplacian()/density
	velocity.dydt_fun = velocity_fun

	return pressure, velocity

def compressible_flow(G: nx.Graph, viscosity=1.0) -> (vertex_pde, edge_pde):
	pass

def no_slip(dG: nx.Graph) -> Callable:
	''' Create no-slip velocity condition graph boundary ''' 
	boundary = set(dG.edges())
	def vel(t, e):
		if e in boundary or (e[1], e[0]) in boundary:
			return 0.
		return None
	return vel

''' Systems ''' 

def fluid_on_grid():
	G = nx.grid_2d_graph(9, 8)
	pressure, velocity = incompressible_flow(G)
	def pressure_values(t, x):
		if x == (2,2):
			return 1.0
		if x == (6,5):
			return -1.0
		return None
	pressure.set_boundary(dirichlet=pressure_values, dynamic=False)
	return pressure, velocity

def fluid_on_circle():
	n = 10
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	pressure, velocity = incompressible_flow(G)
	def pressure_values(t, x):
		if x == 0:
			return 1.0
		if x == n-1:
			return -1.0
		return None
	pressure.set_boundary(dirichlet=pressure_values, dynamic=False)
	return pressure, velocity

def differential_inlets():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4,5,6])
	G.add_edges_from([(1,2),(2,3),(4,3),(2,5),(3,5),(5,6)])
	pressure, velocity = incompressible_flow(G)
	p_vals = {1: 0.2, 4: 0.1, 6: -0.3}
	pressure.set_boundary(dirichlet=dict_fun(p_vals))
	return pressure, velocity

def poiseuille(G: nx.Graph, dG: nx.Graph, dG_L: nx.Graph, dG_R: nx.Graph, p_L=1.0, p_R=-1.0):
	''' Pressure-driven flow by gradient across dG_L -> dG_R ''' 
	pressure, velocity = incompressible_flow(G)
	L_nodes = set(dG_L.nodes())
	R_nodes = set(dG_R.nodes())
	def pressure_values(t, x):
		if x in L_nodes:
			return p_L
		elif x in R_nodes:
			return p_R
		return None
	pressure.set_boundary(dirichlet=pressure_values, dynamic=False)
	velocity.set_boundary(dirichlet=no_slip(dG), dynamic=False)
	return pressure, velocity

def poiseuille_asymmetric(m=10, n=20):
	''' Poiseuille flow with a boundary asymmetry '''
	G = nx.grid_2d_graph(n, m)
	k = 6
	blockage = [
		(k, m-1), (k, m-2),
		(k, m-2), (k, m-3),
		(k, m-2), (k+1, m-2),
		(k, m-3), (k+1, m-3),
		(k+1, m-3), (k+1, m-2),
		(k+1, m-2), (k+1, m-1),
	]
	G.remove_nodes_from(blockage)
	pressure, velocity = incompressible_flow(G)
	def pressure_values(t, x):
		if x[0] == 0: return 1.0
		if x[0] == n-1: return -1.0
		return None
	pressure.set_boundary(dirichlet=pressure_values, dynamic=False)
	velocity.set_boundary(dirichlet=no_slip(G), dynamic=False)
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
	velocity.set_boundary(dirichlet=bc, dynamic=False)
	return pressure, velocity

def von_karman(m=10, n=20):
	G = nx.grid_2d_graph(n, m)
	j, k = 6, int(m/2)
	obstacle = [ # Introduce occlusion
		(j, k-1), (j, k), 
		(j+1, k-1), (j+1, k), 
		(j+2, k-1),
	]
	G.remove_nodes_from(obstacle)
	pressure, velocity = incompressible_flow(G)
	def pressure_values(t, x):
		if x[0] == 0: return 1.0
		if x[0] == n-1: return -1.0
		return None
	pressure.set_boundary(dirichlet=pressure_values, dynamic=False)
	velocity.set_boundary(dirichlet=no_slip(G), dynamic=False)
	return pressure, velocity

def random_graph():
	set_seed(1001)
	n = 30
	eps = 0.3
	G = nx.random_geometric_graph(n, eps)
	pressure, velocity = incompressible_flow(G)
	pressure.set_boundary(dirichlet=dict_fun({4: 1.0, 21: -1.0}))
	return pressure, velocity

def test():
	# G = nx.Graph()
	# G.add_nodes_from([1,2,3,4])
	# G.add_edges_from([(1,2),(2,3),(3,4)])
	# pressure, velocity = incompressible_flow(G)
	# p_vals = {}
	# v_vals = {(1, 2): 1.0, (2, 3): 1.0}
	# pressure.set_boundary(dirichlet=dict_fun(p_vals))
	# velocity.set_boundary(dirichlet=dict_fun(v_vals))

	n = 20
	G = nx.Graph()
	G.add_nodes_from(list(range(n)))
	G.add_edges_from(list(zip(range(n), [n-1] + list(range(n-1)))))
	pressure, velocity = incompressible_flow(G)
	velocity.set_initial(y0=dict_fun({(2,3): 1.0, (3,4): 1.0}, def_val=0.))
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
	# G = nx.triangular_lattice_graph(10, 30)
	# G = nx.hexagonal_lattice_graph(15, 21)
	# dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	# p, v = poiseuille(G, dG, dG_L, dG_R)

	p, v = test()

	# d = v.project(GraphDomain.vertices, lambda v: v.div())
	adv = v.project(GraphDomain.edges, lambda v: v.advect())
	# grad = p.project(GraphDomain.edges, lambda p: p.grad())
	pv = couple(p, v)
	sys = System(pv, {
		'pressure': p,
		'velocity': v,
		# 'div_velocity': d,
		'advection': adv,
		# 'grad': grad,
	})
	# sys.solve_to_disk(10, 1e-2, 'poiseuille_hex')

	''' Load from disk ''' 
	# sys = System.from_disk('poiseuille_hex')
	# p, v, d = sys.observables['pressure'], sys.observables['velocity'], sys.observables['div_velocity']

	renderer = LiveRenderer(sys, [[[[p, v]], [[adv]]]], node_palette=cc.rainbow, node_rng=(-1,1), edge_max=0.3, node_size=0.03)
	renderer.start()
