''' Experimentation / observation ''' 

class TurbulenceObservable(Observable):
	def __init__(self, velocity: edge_gds):
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
	def __init__(self, pressure: node_gds, velocity: edge_gds, **kwargs):
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

# TODO: fix
# def project_cycle_basis(p: gds) -> List[EdgeObservable]:
# 	class ProjObservable(EdgeObservable):
# 		def __init__(self, cycle: list):
# 			self.G = nx.Graph()
# 			nx.add_cycle(self.G, cycle)
# 			self.edge_indices = [p.edges[e] for e in self.G.edges()]
# 			self.orientation = np.array([p.orientation[e] for e in self.G.edges()])
# 		@property
# 		def t(self):
# 			return p.t
# 		@property
# 		def y(self):
# 			return p.y[self.edge_indices] * self.orientation
		
# 	return [ProjObservable(cycle) for cycle in nx.cycle_basis(p.G)]

''' Other derived observables for measurement ''' 

class MetricsObservable(Observable):
	''' Observable for measuring scalar derived quantities ''' 
	def __init__(self, base: Observable, metrics: Dict):
		self.base = base
		self.metrics = metrics
		X = dict(zip(metrics.keys(), count()))
		super().__init__(self, X)

	@property 
	def t(self):
		return self.base.t

	@property 
	def y(self):
		''' Call in order to update metrics. TODO: brittle? ''' 
		self.metrics = self.calculate(self.base.y, self.metrics)
		return self.metrics

	@abstractmethod
	def calculate(self, y: Any, metrics: Dict):
		''' Override to calculate metrics set ''' 
		pass


