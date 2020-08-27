import numpy as np
from typing import List, Tuple, Any, Dict, NewType
import colorcet as cc
import shortuuid
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import zmq
import time
from multiprocessing import Process

from bokeh.plotting import figure, from_networkx
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, HoverTool, Arrow, VeeHead
from bokeh.models.glyphs import Oval, MultiLine
from bokeh.transform import linear_cmap
from bokeh.command.util import build_single_handler_applications
from bokeh.server.server import Server
from bokeh.util.browser import view
from tornado.ioloop import IOLoop

from gpde import *
from gpde.utils.zmq import *

''' Common types ''' 

Canvas = List[List[List[List[Observable]]]]
Plot = Any
PlotID = NewType('PlotID', str)

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

	def draw_plots(self, root):
		''' Draw plots to bokeh element ''' 
		rows = []
		for i in range(len(self.canvas)):
			cols = []
			for j in range(len(self.canvas[i])):
				if len(self.canvas[i][j]) == 1:
					cols.append(self.create_plot(self.canvas[i][j][0]))
				else:
					subplots = []
					for k in range(len(self.canvas[i][j])):
						subplots.append(self.create_plot(self.canvas[i][j][k]))
					nsubcols = int(np.sqrt(len(subplots)))
					cols.append(gridplot(subplots, ncols=nsubcols, sizing_mode='scale_both'))
			rows.append(row(cols, sizing_mode='scale_both'))
		root.children.append(column(rows, sizing_mode='scale_both'))

	def create_plot(self, items: List[Observable]):
		assert all([obs.G is items[0].G for obs in items]), 'Co-rendered observables must use the same graph'
		G = nx.convert_node_labels_to_integers(items[0].G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
		layout = self.layout_func(G)
		def helper(obs: Observable, plot=None):
			if plot is None:
				plot = figure(x_range=(-1.1,1.1), y_range=(-1.1,1.1), tooltips=[])
				plot.axis.visible = None
				plot.xgrid.grid_line_color = None
				plot.ygrid.grid_line_color = None
				renderer = from_networkx(G, layout)
				plot.renderers.append(renderer)
			# Domain-specific rendering
			desc = 'value' # TODO
			if isinstance(obs, VertexObservable):
				plot.renderers[0].node_renderer.data_source.data['node'] = list(G.nodes())
				plot.renderers[0].node_renderer.data_source.data['value'] = obs.y 
				plot.renderers[0].node_renderer.glyph = Oval(height=0.04, width=0.04, fill_color=linear_cmap('value', self.palette, self.lo, self.hi))
				if self.show_bar:
					cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.palette, low=self.lo, high=self.hi), ticker=BasicTicker(), title=desc)
					plot.add_layout(cbar, 'right')
				plot.add_tools(HoverTool(tooltips=[('value', '@value'), ('node', '@node')]))
			elif isinstance(obs, EdgeObservable):
				source = plot.renderers[0].edge_renderer.data_source
				obs.layout_data = pd.DataFrame(
					[[layout[x1][0], layout[x1][1], layout[x2][0], layout[x2][1]] for (x1, x2) in G.edges()],
					columns=['x1', 'y1', 'x2', 'y2']
				)
				source.data['value'] = np.ones(len(obs.y))
				source.data['x1'], source.data['x2'], source.data['y1'], source.data['y2'] = obs.layout_data['x1'], obs.layout_data['x2'], obs.layout_data['y1'], obs.layout_data['y2']
				self.draw_arrows(source, obs)
				if self.show_bar:
					cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.palette, low=self.lo, high=self.hi), ticker=BasicTicker(), title=desc)
					plot.add_layout(cbar, 'right')
				arrows = Arrow(
					end=VeeHead(size=8), 
					x_start='x1', y_start='y1', x_end='x2', y_end='y2', line_width=0, 
					source=source
				)
				plot.add_layout(arrows)
			return plot
		
		plot = None
		plot_id = shortuuid.uuid()
		for obs in items:
			plot = helper(obs, plot)
			obs.plot_id = plot_id
			self.plots[plot_id] = plot
		return plot

	def draw(self):
		for obs in self.observables:
			plot = self.plots[obs.plot_id]
			if isinstance(obs, VertexObservable):
				plot.renderers[0].node_renderer.data_source.data['value'] = obs.y
			elif isinstance(obs, EdgeObservable):
				# TODO: render edge direction using: https://discourse.bokeh.org/t/hover-over-tooltips-on-network-edges/2439/7
				self.draw_arrows(plot.renderers[0].edge_renderer.data_source, obs)

	def draw_arrows(self, source, obs):
		idx = np.sign(source.data['value']) != obs.y
		tmp = source.data['x1']
		source.data['x1'][idx] = source.data['x2'][idx] 
		source.data['x2'][idx] = tmp[idx] 
		tmp = source.data['y1']
		source.data['y1'][idx] = source.data['y2'][idx] 
		source.data['y2'][idx] = tmp[idx] 
		source.data['value'] = obs.y


''' Layout-specific renderers ''' 

class SingleRenderer(Renderer):
	''' Render all observables in the same plot ''' 
	def setup_canvas(self):
		return [[[self.observables]]]

class GridRenderer(Renderer):
	''' Render all observables separately as items on a grid ''' 

	def __init__(self, *args, ncols=2, **kwargs):
		self.ncols = ncols
		super().__init__(*args, **kwargs)

	def setup_canvas(self):
		canvas = []
		for i, obs in enumerate(self.observables):
			if i % self.ncols == 0:
				canvas.append([])
			row = canvas[-1]
			row.append([(obs,)])
		return canvas

''' Entry point ''' 

host = 'localhost'
port = 8080

def render_bokeh(renderer: Renderer):
	''' Render as a Bokeh web app ''' 
	path = str(Path(__file__).parent / 'bokeh_server.py')
	proc = Process(target=start_server, args=(path, host, port))
	proc.start()
	print('Server started')
	ctx, tx = ipc_tx()

	try:
		print('Waiting for server to initialize...')
		tx({'tag': 'init', 'renderer': wire_pickle(renderer)})
		print('Done.')
		while True: 
			time.sleep(1) # Let bokeh continue to handle interactivity while we wait
	finally:
		ctx.destroy()
		proc.terminate()

''' Helpers ''' 

def start_server(filepath: str, host: str, port: int):
	files = [filepath]
	argvs = {}
	urls = []
	for f in files:
		argvs[f]=None
		urls.append(f.split('/')[-1].split('.')[0])
	io_loop = IOLoop.instance()
	apps = build_single_handler_applications(files,argvs)
	kwags = {
		'io_loop':io_loop,
		'generade_session_ids':True,
		'redirect_root':True,
		'use_x_headers':False,
		'secret_key':None,
		'num_procs':1,
		'host':['%s:%d'%(host, port)],
		'sign_sessions':False,
		'develop':False,
		'port':port, 
		'use_index':True
	}
	srv = Server(apps,**kwags)
	io_loop.add_callback(view, 'http://{}:{}'.format(host, port))
	io_loop.start()