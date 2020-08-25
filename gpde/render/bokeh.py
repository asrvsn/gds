import numpy as np
from typing import List, Tuple, Any, Dict
import colorcet as cc
import shortuuid
import pandas as pd
from abc import ABC, abstractmethod

from bokeh.plotting import figure, from_networkx
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, HoverTool, Arrow, VeeHead
from bokeh.models.glyphs import Oval, MultiLine
from bokeh.transform import linear_cmap

from gpde.core import *

''' Common types ''' 

Canvas = List[List[List[Tuple[Observable]]]]
Plot = Any
PlotID = Newtype('PlotID', str)

''' Classes ''' 

class Renderer(ABC):
	def __init__(self, sys: System, palette=cc.fire, lo=0., hi=1., layout_func=None, n_spring_iters=500, show_bar=True):
		self.integrator = sys[0]
		self.observables = sys[1]
		self.canvas: Canvas = self.setup_canvas()
		self.plots: Dict[PlotID, Plot] = dict()
		self.palette = palette
		self.lo = lo
		self.hi = hi
		self.show_bar=show_bar
		if layout_func is None:
			self.layout_func = lambda G: nx.spring_layout(G, scale=0.9, center=(0,0), iterations=n_spring_iters, seed=1)
		else:
			self.layout_func = layout_func

	@abstractmethod
	def setup_canvas(self) -> Canvas:
		pass

	def setup_plots(self, root):
		''' Draw plots to bokeh element ''' 
		rows = []
		for i in range(len(self.canvas)):
			cols = []
			for j in range(len(self.canvas[i])):
				if len(self.canvas[i][j]) == 1:
					cols.append(create_plot(self.canvas[i][j][0]))
				else:
					subplots = []
					for k in range(len(self.canvas[i][j])):
						subplots.append(create_plot(self.canvas[i][j][k]))
					nsubcols = int(np.sqrt(len(subplots)))
					cols.append(gridplot(subplots, ncols=nsubcols, sizing_mode='scale_both'))
			rows.append(row(cols, sizing_mode='scale_both'))
		root.children.append(column(rows, sizing_mode='scale_both'))

	def create_plot(self, items: Tuple[Observable]):
		def helper(obs: Observable, plot=None):
			if plot is None:
				G = nx.convert_node_labels_to_integers(obs.G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
				layout = self.layout_func(G)
				plot = figure(x_range=(-1.1,1.1), y_range=(-1.1,1.1), tooltips=[])
				plot.axis.visible = None
				plot.xgrid.grid_line_color = None
				plot.ygrid.grid_line_color = None
				renderer = from_networkx(G, layout)
				plot.renderers.append(renderer)
			# Domain-specific rendering
			if isinstance(obs, VertexObservable):
				plot.renderers[0].node_renderer.data_source.data['node'] = list(G.nodes())
				plot.renderers[0].node_renderer.data_source.data['value'] = obs.y 
				plot.renderers[0].node_renderer.glyph = Oval(height=0.08, width=0.08, fill_color=linear_cmap('value', self.palette, self.lo, self.hi))
				if self.show_bar:
					cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.palette, low=self.lo, high=self.hi), ticker=BasicTicker(), title=self.desc)
					plot.add_layout(cbar, 'right')
				plot.add_tools(HoverTool(tooltips=[('value', '@value'), ('node', '@node')]))
			elif isinstance(obs, EdgeObservable):
				plot.renderers[0].edge_renderer.data_source.data['value'] = obs.y
				layout_coords = pd.DataFrame(
					[[layout[x1][0], layout[x1][1], layout[x2][0], layout[x2][1]] for (x1, x2) in G.edges()],
					columns=['x_start', 'y_start', 'x_end', 'y_end']
				)
				layout_coords['x_end'] = (layout_coords['x_end'] - layout_coords['x_start']) / 2 + layout_coords['x_start']
				layout_coords['y_end'] = (layout_coords['y_end'] - layout_coords['y_start']) / 2 + layout_coords['y_start']
				plot.renderers[0].edge_renderer.data_source.data['x_start'] = layout_coords['x_start']
				plot.renderers[0].edge_renderer.data_source.data['y_start'] = layout_coords['y_start']
				plot.renderers[0].edge_renderer.data_source.data['x_end'] = layout_coords['x_end']
				plot.renderers[0].edge_renderer.data_source.data['y_end'] = layout_coords['y_end']
				plot.renderers[0].edge_renderer.glyph = MultiLine(line_color=linear_cmap('value', self.palette, self.lo, self.hi), line_width=5)
				if self.show_bar:
					cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.palette, low=self.lo, high=self.hi), ticker=BasicTicker(), title='value')
					plot.add_layout(cbar, 'right')
				arrows = Arrow(
					end=VeeHead(size=8), 
					x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end', line_width=0, 
					source=plot.renderers[0].edge_renderer.data_source
				)
				plot.add_layout(arrows)
			return plot
		
		plot = None
		plot_id = shortuuid.uuid()
		for obs in items:
			plot = helper(obs, plot)
			obs.plot_id = plot_id
		return plot

	def draw(self):
		for obs in self.observables:
			plot = self.plots[obs.plot_id]
			if isinstance(obs, VertexObservable):
				plot.renderers[0].node_renderer.data_source.data['value'] = obs.y
			elif isinstance(obs, EdgeObservable):
				# TODO: render edge direction using: https://discourse.bokeh.org/t/hover-over-tooltips-on-network-edges/2439/7
				plot.renderers[0].edge_renderer.data_source.data['value'] = obs.y


''' Layout-specific renderers ''' 

class GridRenderer(Renderer):
	''' Render all observables separately as items on a grid ''' 

	def __init__(*args, ncols=2, **kwargs):
		super().__init__(*args, **kwargs)
		self.ncols = ncols

	def setup_canvas(self):
		canvas = []
		for i, obs in enumerate(self.observables):
			if i % self.ncols == 0:
				canvas.append([])
			row = canvas[-1]
			row.append([(obs,)])
		return canvas

''' Entry point ''' 

def render_bokeh(renderer: Renderer):
	''' Render as a Bokeh web app ''' 
	pass