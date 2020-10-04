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
import networkx as nx
import os.path
import cloudpickle

from bokeh.plotting import figure, from_networkx
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, HoverTool, Arrow, VeeHead, ColumnDataSource
from bokeh.models.glyphs import Oval, MultiLine, Patches
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
	def __init__(self, 
				canvas: Canvas,
				node_palette=cc.fire, edge_palette=cc.fire, layout_func=None, n_spring_iters=500, dim=2, 
				node_rng=(0., 1.), edge_rng=(0., 1.), edge_max=0.25, colorbars=True, 
				node_size=0.04
			):
		self.canvas: Canvas = canvas
		self.plots: Dict[PlotID, Plot] = dict()
		self.node_palette = node_palette
		self.edge_palette = edge_palette
		self.node_rng = node_rng
		self.edge_rng = edge_rng
		self.colorbars = colorbars
		self.edge_max = edge_max
		self.node_size = node_size
		if layout_func is None:
			def func(G):
				pos_attr = nx.get_node_attributes(G, 'pos')
				if len(pos_attr) > 0: # G already has self-defined positions
					return {k: (np.array(v) - 0.5)*2 for k, v in pos_attr.items()}
				else:
					return nx.spring_layout(G, scale=0.9, center=(0,0), iterations=n_spring_iters, seed=1, dim=dim)
			self.layout_func = func
		else:
			self.layout_func = layout_func

	''' Overrides ''' 

	@abstractmethod
	def step(self, dt: float): 
		''' Draw data to plots '''
		pass

	@property
	@abstractmethod
	def t(self) -> float:
		''' Get current time ''' 
		pass

	''' API ''' 

	def start(self):
		''' entry point ''' 
		render_bokeh(self)

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
				# plot.axis.visible = None
				# plot.xgrid.grid_line_color = None
				# plot.ygrid.grid_line_color = None
				renderer = from_networkx(G, layout)
				plot.renderers.append(renderer)
				plot.add_tools(HoverTool(tooltips=[('value', '@value'), ('node', '@node'), ('edge', '@edge')]))
			# Domain-specific rendering
			if isinstance(obs, GraphObservable):
				if obs.Gd is GraphDomain.vertices: 
					plot.renderers[0].node_renderer.data_source.data['node'] = list(map(str, items[0].G.nodes()))
					plot.renderers[0].node_renderer.data_source.data['value'] = obs.y 
					if isinstance(obs, gpde):
						plot.renderers[0].node_renderer.data_source.data['thickness'] = [3 if (x in obs.dirichlet_X or x in obs.neumann_X) else 1 for x in obs.X] 
						plot.renderers[0].node_renderer.glyph = Oval(height=self.node_size, width=self.node_size, fill_color=linear_cmap('value', self.node_palette, self.node_rng[0], self.node_rng[1]), line_width='thickness')
					else:
						plot.renderers[0].node_renderer.glyph = Oval(height=self.node_size, width=self.node_size, fill_color=linear_cmap('value', self.node_palette, self.node_rng[0], self.node_rng[1]))
					if self.colorbars:
						cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.node_palette, low=self.node_rng[0], high=self.node_rng[1]), ticker=BasicTicker(), title='node')
						plot.add_layout(cbar, 'right')
				elif obs.Gd is GraphDomain.edges:
					self.prep_layout_data(obs, G, layout)
					obs.arr_source.data['edge'] = list(map(str, items[0].G.edges()))
					self.draw_arrows(obs, obs.y)
					if self.colorbars:
						cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.edge_palette, low=self.edge_rng[0], high=self.edge_rng[1]), ticker=BasicTicker(), title='edge')
						plot.add_layout(cbar, 'right')
					arrows = Patches(xs='xs', ys='ys', fill_color=linear_cmap('value', self.edge_palette, low=self.edge_rng[0], high=self.edge_rng[1]))
					plot.add_glyph(obs.arr_source, arrows)
				else:
					raise Exception('unknown graph domain.')
			return plot
		
		plot = None
		plot_id = shortuuid.uuid()
		for obs in items:
			plot = helper(obs, plot)
			obs.plot_id = plot_id
			self.plots[plot_id] = plot
		return plot

	def prep_layout_data(self, obs, G, layout):
		data = pd.DataFrame(
			[[layout[x1][0], layout[x1][1], layout[x2][0], layout[x2][1]] for (x1, x2) in G.edges()],
			columns=['x1', 'y1', 'x2', 'y2']
		)
		data['dx'] = data['x2'] - data['x1']
		data['dx_dir'] = np.sign(data['dx'])
		data['dy'] = data['y2'] - data['y1']
		data['x_mid'] = data['x1'] + data['dx'] / 2
		data['y_mid'] = data['y1'] + data['dy'] / 2
		data['m_norm'] = np.sqrt(data['dx']**2 + data['dy']**2)
		data['dx'] /= data['m_norm'] # TODO: possible division by 0
		data['dy'] /= data['m_norm'] # TODO: possible division by 0
		data['m'] = data['dy'] / data['dx'] # TODO: possible division by 0
		obs.layout = data
		obs.arr_source = ColumnDataSource()

	def draw_arrows(self, obs, y):
		h = 0.1
		w = 0.1
		absy = np.abs(y)
		magn = np.clip(np.log(1 + absy), a_min=None, a_max=self.edge_max)
		p1x = obs.layout['x_mid']
		p1y = obs.layout['y_mid']
		dx = -np.sign(obs.y) * magn * obs.layout['dx_dir'] * h / np.sqrt(obs.layout['m'] ** 2 + 1)
		dy = obs.layout['m'] * dx
		p2x = -obs.layout['dy'] * magn * w/2 + p1x + dx
		p2y = obs.layout['dx'] * magn * w/2 + p1y + dy
		p3x = obs.layout['dy'] * magn * w/2 + p1x + dx
		p3y = -obs.layout['dx'] * magn * w/2 + p1y + dy
		obs.arr_source.data['xs'] = np.stack((p1x, p2x, p3x), axis=1).tolist()
		obs.arr_source.data['ys'] = np.stack((p1y, p2y, p3y), axis=1).tolist()
		obs.arr_source.data['value'] = absy


''' Derivations ''' 

class LiveRenderer(Renderer):
	''' Simultaneously solves & renders the system ''' 
	def __init__(self, sys: System, *args, **kwargs):
		self.system = sys
		self.integrator = sys.integrator
		self.observables = list(sys.observables.values())
		super().__init__(*args, **kwargs)

	def step(self, dt: float):
		self.integrator.step(dt)
		self.draw()

	def reset(self):
		self.integrator.reset()
		self.draw()

	def draw(self):
		for obs in self.observables:
			plot = self.plots[obs.plot_id]
			if obs.Gd is GraphDomain.vertices:
				self.plots[obs.plot_id].renderers[0].node_renderer.data_source.data['value'] = obs.y
			elif obs.Gd is GraphDomain.edges:
				self.draw_arrows(obs, obs.y)

	@property
	def t(self):
		return self.integrator.t

class StaticRenderer(Renderer):
	''' Reads a solution from disk & renders it ''' 
	def __init__(self, path: str, *args, **kwargs):
		assert os.path.isdir(path), 'The given path does not exist'
		with open(f'{path}/system.pkl', 'rb') as f:
			self.system = cloudpickle.load(f)
		self.data = dict()
		for name in self.system.observables.keys():
			self.data[name] = hkl.load(f'{path}/{name}.hkl')
		self._t = 0.
		self._i = 0
		super().__init__(*args, **kwargs)

	def step(self, dt: float):
		T = self._t + dt
		while self._t < t:
			for name, obs in self.system.observables.items():
				if obs.Gd is GraphDomain.vertices:
					self.plots[obs.plot_id].renderers[0].node_renderer.data_source.data['value'] = self.data[name][self._i]
				elif obs.Gd is GraphDomain.edges:
					self.draw_arrows(obs, self.data[name][self._i])
			self._t += self.system.dt
			self._i += 1

	@property
	def t(self):
		return self._t

''' Layout creators ''' 

def single_canvas(observables: List[Observable]) -> Canvas:
	''' Render all observables in the same plot ''' 
	return [[[observables]]]

def grid_canvas(observables: List[Observable], ncols: int=2) -> Canvas:
	''' Render all observables separately as items on a grid ''' 
	canvas = []
	for i, obs in enumerate(observables):
		if i % ncols == 0:
			canvas.append([])
		row = canvas[-1]
		row.append([(obs,)])
	return canvas

''' Server ''' 

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