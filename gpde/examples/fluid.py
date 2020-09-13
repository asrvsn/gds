import networkx as nx
import numpy as np
import pdb
from itertools import count

from gpde.utils import set_seed
from gpde.core import *
from gpde.render.bokeh import *

''' Definitions ''' 

def velocity_eq(G: nx.Graph, pressure: vertex_pde, kinematic_viscosity: float=1.0):
	def f(t, self):
		return -self.advect_self() - pressure.grad() + kinematic_viscosity * self.helmholtzian()
	return edge_pde(G, f)

def fluid_on_grid():
	n = 8
	G = nx.grid_2d_graph(n, n)
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	def pressure_values(x):
		if x == (3,3):
			return 1.0
		if x == (5,5):
			return -1.0
		return 0.
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
	return pressure, velocity

def differential_inlets():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4,5,6])
	G.add_edges_from([(1,2),(2,3),(4,3),(2,5),(3,5),(5,6)])
	def pressure_values(x):
		if x == 1: return 1.0
		if x == 4: return 0.5
		if x == 6: return -0.5
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
	return pressure, velocity

def poiseuille():
	G = nx.grid_2d_graph(10, 5)
	def pressure_values(x):
		if x[0] == 0: return 1.0
		if x[0] == 9: return -1.0
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
	def no_slip(t, x):
		if x[0][1] == x[1][1] == 0 or x[0][1] == x[1][1] == 4:
			return 0.
		return None
	velocity.set_boundary(dirichlet=no_slip, dynamic=False)
	return pressure, velocity

def fluid_on_sphere():
	pass

def von_karman():
	w, h = 20, 10
	G = nx.grid_2d_graph(w, h)
	obstacle = [ # Introduce occlusion
		(6, 4), (6, 5), 
		(7, 4), (7, 5), 
		(8, 4),
	]
	G.remove_nodes_from(obstacle)
	def pressure_values(x):
		if x[0] == 0: return 1.0
		if x[0] == w-1: return -1.0
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
	return pressure, velocity

def random_graph():
	set_seed(1001)
	n = 30
	eps = 0.3
	G = nx.random_geometric_graph(n, eps)
	def pressure_values(x):
		if x == 5: return 1.0
		return 0.
	pressure = vertex_pde(G, lambda t, self: np.zeros(len(self)))
	pressure.set_initial(y0=pressure_values)
	velocity = velocity_eq(G, pressure)
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
		self.rendered = set({'v_sigma',})
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
		print(ret)
		return ret
	
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
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
			plot = figure(title='Turbulence metrics', y_range=cats)
			plot.x_range.follow = 'end'
			plot.x_range.follow_interval = 5.0
			plot.x_range.range_padding = 0
			for i, cat in enumerate(reversed(cats)):
				plot.patch('t', cat, color=cc.glasbey[i], alpha=0.6, line_color='black', source=src)
			plot.outline_line_color = None
			plot.ygrid.grid_line_color = None
			plot.xgrid.grid_line_color = '#dddddd'
			plot.xgrid.ticker = plot.xaxis.ticker
			self.turbulence.src = src
			return plot
		else:
			return super().create_plot(items)

	def draw(self):
		self.turbulence.src.stream(self.turbulence.observe(), 100)
		super().draw()

if __name__ == '__main__':
	p, v = differential_inlets()

	# sys = couple(p, v)
	# renderer = SingleRenderer(sys, node_rng=(-1,1))
	# cycles = project_cycle_basis(v)
	# renderer = CustomRenderer(sys[0], [[[[p, v]], [[c] for c in cycles]]], node_rng=(-1,1), colorbars=False)

	renderer = FluidRenderer(p, v, node_rng=(-1, 1))
	render_bokeh(renderer)
