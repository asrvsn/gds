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
import random

from bokeh.core.properties import field
from bokeh.plotting import figure, from_networkx
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, HoverTool, Arrow, VeeHead, ColumnDataSource
from bokeh.models.glyphs import Ellipse, MultiLine, Patches
from bokeh.transform import linear_cmap
from bokeh.command.util import build_single_handler_applications
from bokeh.server.server import Server
from bokeh.util.browser import view
from bokeh.io import export_png
from bokeh.models.widgets import Div
from tornado.ioloop import IOLoop

from gds import *
from gds.utils.zmq import *
from gds.utils import now
from .base import *

''' Classes ''' 

class Renderer(ABC):
	def __init__(self, 
				canvas: Canvas,
				node_palette=cc.fire, edge_palette=cc.fire, face_palette=cc.fire, layout_func=None, n_spring_iters=500, dim=2, 
				node_rng=(0., 1.), edge_rng=(0., 1.), face_rng=(0., 1.), edge_max=0.2, colorbars=True, 
				node_size=0.06, plot_width=700, plot_height=750, dynamic_ranges=False,
				x_rng=(-1.1,1.1), y_rng=(-1.1,1.1),
				edge_colors=False, min_rng_size=0,
				title=None, plot_titles=True,
			):
		self.canvas: Canvas = canvas
		self.plots: Dict[PlotID, Plot] = dict()
		self.node_cmaps: Dict[PlotID, ColorBar] = dict()
		self.edge_cmaps: Dict[PlotID, ColorBar] = dict()
		self.face_cmaps: Dict[PlotID, ColorBar] = dict()
		self.node_palette = node_palette
		self.edge_palette = edge_palette
		self.face_palette = face_palette
		self.node_rng = node_rng
		self.edge_rng = edge_rng
		self.face_rng = face_rng
		self.colorbars = colorbars
		self.edge_max = edge_max
		self.node_size = node_size
		self.plot_width = plot_width
		self.plot_height = plot_height
		self.dynamic_ranges = dynamic_ranges
		self.x_rng, self.y_rng = x_rng, y_rng
		self.edge_colors = edge_colors
		self.min_rng_size = min_rng_size
		self.title = title
		self.plot_titles = plot_titles
		if layout_func is None:
			def func(G):
				pos = nx.get_node_attributes(G, 'pos')
				if len(pos) > 0: # G already has self-defined positions
					xmin = min(map(lambda a: a[0], pos.values()))
					xmax = max(map(lambda a: a[0], pos.values()))
					ymin = min(map(lambda a: a[1], pos.values()))
					ymax = max(map(lambda a: a[1], pos.values()))
					vmin, vmax = np.array([xmin, ymin]), np.array([xmax, ymax])
					return {k: ((np.array(v) - vmin)/(vmax - vmin) - 0.5)*2 for k, v in pos.items()}
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
		if self.title != None:
			rows.append(Div(text=self.title, style={'font-size':'200%'}))
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
					cols.append(gridplot(subplots, ncols=nsubcols))
			rows.append(row(cols))
		self.root_plot = column(rows)
		root.children.append(self.root_plot)

	def create_plot(self, items: List[Observable]):
		assert all([obs.G is items[0].G for obs in items]), 'Co-rendered observables must use the same graph'
		orig_G = items[0].G
		orig_layout = self.layout_func(orig_G)
		G = nx.convert_node_labels_to_integers(orig_G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
		G = clear_attributes(G)
		layout = {i: orig_layout[n] for i, n in enumerate(orig_G.nodes())}
		def helper(obs: Observable, plot=None):
			if plot is None:
				plot = figure(x_range=self.x_rng, y_range=self.y_rng, tooltips=[], width=self.plot_width, height=self.plot_height) 
				plot.axis.visible = None
				plot.xgrid.grid_line_color = None
				plot.ygrid.grid_line_color = None
				renderer = from_networkx(G, layout)
				plot.renderers.append(renderer)
				plot.add_tools(HoverTool(tooltips=[('value', '@value'), ('node', '@node'), ('edge', '@edge')]))
				plot.toolbar_location = None
			# Domain-specific rendering
			if isinstance(obs, GraphObservable):
				if obs.Gd is GraphDomain.nodes: 
					plot.renderers[0].node_renderer.data_source.data['node'] = list(map(str, items[0].G.nodes()))
					plot.renderers[0].node_renderer.data_source.data['value'] = obs.y 
					cmap = LinearColorMapper(palette=self.node_palette, low=self.node_rng[0], high=self.node_rng[1])
					self.node_cmaps[obs.plot_id] = cmap
					if isinstance(obs, gds):
						plot.renderers[0].node_renderer.data_source.data['thickness'] = [3 if (x in obs.X_dirichlet or x in obs.X_neumann) else 1 for x in obs.X] 
						plot.renderers[0].node_renderer.glyph = Ellipse(height=self.node_size, width=self.node_size, fill_color=field('value', cmap), line_width='thickness')
					else:
						plot.renderers[0].node_renderer.glyph = Ellipse(height=self.node_size, width=self.node_size, fill_color=field('value', cmap))
					if self.colorbars:
						cbar = ColorBar(color_mapper=cmap, ticker=BasicTicker(), title='node')
						plot.add_layout(cbar, 'right')
				elif obs.Gd is GraphDomain.edges:
					self.prep_layout_data(obs, G, layout)
					obs.arr_source.data['edge'] = list(map(str, items[0].G.edges()))
					self.draw_arrows(obs, obs.y)
					plot.renderers[0].edge_renderer.data_source.data['value'] = obs.arr_source.data['value']
					cmap = LinearColorMapper(palette=self.edge_palette, low=self.edge_rng[0], high=self.edge_rng[1])
					self.edge_cmaps[obs.plot_id] = cmap
					arrows = Patches(xs='xs', ys='ys', fill_color=field('value', cmap))
					plot.add_glyph(obs.arr_source, arrows)
					if self.colorbars:
						cbar = ColorBar(color_mapper=cmap, ticker=BasicTicker(), title='edge')
						plot.add_layout(cbar, 'right')
					if isinstance(obs, gds):
						if self.edge_colors:
							plot.renderers[0].edge_renderer.glyph = MultiLine(line_width=5, line_color=field('value', cmap))
						else:
							plot.renderers[0].edge_renderer.data_source.data['thickness'] = [3 if (x in obs.X_dirichlet or x in obs.X_neumann) else 1 for x in obs.X] 
							plot.renderers[0].edge_renderer.glyph = MultiLine(line_width='thickness')
				elif obs.Gd is GraphDomain.faces:
					cmap = LinearColorMapper(palette=self.face_palette, low=self.face_rng[0], high=self.face_rng[1])
					self.face_cmaps[obs.plot_id] = cmap
					obs.face_source = ColumnDataSource()
					xs = [[orig_layout[n][0] for n in f] for f in obs.faces]
					ys = [[orig_layout[n][1] for n in f] for f in obs.faces]
					obs.face_source.data['xs'] = xs
					obs.face_source.data['ys'] = ys
					obs.face_source.data['value'] = np.zeros(obs.ndim)
					faces = Patches(xs='xs', ys='ys', fill_color=field('value', cmap), line_color='#FFFFFF', line_width=2)
					plot.add_glyph(obs.face_source, faces)
					if self.colorbars:
						cbar = ColorBar(color_mapper=cmap, ticker=BasicTicker(), title='face')
						plot.add_layout(cbar, 'right')
				else:
					raise Exception('unknown graph domain.')
			return plot
		
		plot = None
		plot_id = shortuuid.uuid()
		for obs in items:
			obs.plot_id = plot_id
			plot = helper(obs, plot)
		self.plots[plot_id] = plot
		return plot

	def prep_layout_data(self, obs, G, layout):
		data = pd.DataFrame(
			[[layout[x1][0], layout[x1][1], layout[x2][0], layout[x2][1]] for (x1, x2) in G.edges()],
			columns=['x1', 'y1', 'x2', 'y2']
		)
		dx = data['x2'] - data['x1']
		dx[dx >= 0] = np.maximum(1e-3, dx[dx >= 0])
		dx[dx < 0] = np.minimum(-1e-3, dx[dx < 0])
		data['dx'] = dx
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
		scaler = lambda x: x
		if self.dynamic_ranges:
			magn = (self.edge_max / max(scaler(absy).max(), 1e-6)) * scaler(absy)
		else:
			# TODO: cleanup
			magn = np.clip(scaler(absy), a_min=None, a_max=self.edge_max)
		dx = -np.sign(obs.y) * magn * obs.layout['dx_dir'] * h / np.sqrt(obs.layout['m'] ** 2 + 1)
		dy = obs.layout['m'] * dx
		p1x = obs.layout['x_mid'] - dx/2
		p1y = obs.layout['y_mid'] - dy/2
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
	def __init__(self, sys: System, **kwargs):
		self.system = sys
		self.stepper = sys.stepper
		self.stepper.step(0) # Uncover any immediate issues at construction

		self.rec_name = None
		self.rec_ctr = None

		self.observables = list(sys.observables.values())
		if 'canvas' in kwargs:
			canvas = kwargs['canvas']
			del kwargs['canvas']
		else:
			canvas = sys.arrange()
		super().__init__(canvas, **kwargs)

	def draw_plots(self, root):
		super().draw_plots(root)
		if self.plot_titles:
			for plot_id, plot in self.plots.items():
				names = []
				for name, obs in self.system.observables.items():
					if hasattr(obs, 'plot_id') and obs.plot_id == plot_id:
						names.append(name) # Somewhat hacky
				plot.title.text = ','.join(names)

	def step(self, dt: float):
		self.stepper.step(dt)
		self.draw()

	def reset(self):
		self.stepper.reset()
		self.draw()

	def draw(self):
		for obs in self.observables:
			if hasattr(obs, 'plot_id'):
				plot = self.plots[obs.plot_id]
				if obs.Gd is GraphDomain.nodes:
					self.plots[obs.plot_id].renderers[0].node_renderer.data_source.data['value'] = obs.y
					if self.dynamic_ranges:
						lo, hi = obs.y.min(), obs.y.max()
						mid = (lo+hi)/2
						lo, hi = min(lo, mid-self.min_rng_size/2), max(hi, mid+self.min_rng_size/2)
						self.node_cmaps[obs.plot_id].low = lo
						self.node_cmaps[obs.plot_id].high = hi
				elif obs.Gd is GraphDomain.edges:
					self.draw_arrows(obs, obs.y)
					self.plots[obs.plot_id].renderers[0].edge_renderer.data_source.data['value'] = obs.arr_source.data['value']
					if self.dynamic_ranges:
						lo, hi = obs.arr_source.data['value'].min(), obs.arr_source.data['value'].max()
						mid = (lo+hi)/2
						lo, hi = min(lo, mid-self.min_rng_size/2), max(hi, mid+self.min_rng_size/2)
						self.edge_cmaps[obs.plot_id].low = lo
						self.edge_cmaps[obs.plot_id].high = hi
				elif obs.Gd is GraphDomain.faces:
					obs.face_source.data['value'] = obs.y
					if self.dynamic_ranges:
						lo, hi = obs.y.min(), obs.y.max()
						mid = (lo+hi)/2
						lo, hi = min(lo, mid-self.min_rng_size/2), max(hi, mid+self.min_rng_size/2)
						self.face_cmaps[obs.plot_id].low = lo
						self.face_cmaps[obs.plot_id].high = hi
		if self.rec_name != None:
			self.dump_frame()

	def start_recording(self):
		self.rec_ctr = 0
		if not os.path.exists('recordings'):
			os.makedirs('recordings')
		self.rec_name = f'recordings/{now().strftime("%m-%d-%y %H:%M:%S")}'
		os.makedirs(self.rec_name)
		self.dump_frame()

	def dump_frame(self):
		export_png(self.root_plot, filename=f'{self.rec_name}/{self.rec_ctr}.png', timeout=10)
		self.rec_ctr += 1

	def stop_recording(self):
		self.rec_name = None
		self.rec_ctr = None

	@property
	def t(self):
		return self.stepper.t

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
				if obs.Gd is GraphDomain.nodes:
					self.plots[obs.plot_id].renderers[0].node_renderer.data_source.data['value'] = self.data[name][self._i]
				elif obs.Gd is GraphDomain.edges:
					self.draw_arrows(obs, self.data[name][self._i])
			self._t += self.system.dt
			self._i += 1

	@property
	def t(self):
		return self._t

''' Server ''' 

host = 'localhost'
port = random.randint(1000,10000)

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
