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
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, HoverTool, Arrow, VeeHead, ColumnDataSource, Arc
from bokeh.models.glyphs import Ellipse, MultiLine, Patches, Line
from bokeh.transform import linear_cmap
from bokeh.command.util import build_single_handler_applications
from bokeh.server.server import Server
from bokeh.util.browser import view
from bokeh.io import export_png
from bokeh.models.widgets import Div
from tornado.ioloop import IOLoop
from selenium import webdriver

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
				node_size=0.06, plot_width=700, plot_height=750, dynamic_ranges=False, range_padding=0.,
				x_rng=(-1.1,1.1), y_rng=(-1.1,1.1),
				edge_colors=False, min_rng_size=1e-6,
				title=None, plot_titles=True,
				face_orientations=True,
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
		self.range_padding = range_padding
		self.x_rng, self.y_rng = x_rng, y_rng
		self.edge_colors = edge_colors
		self.min_rng_size = min_rng_size
		self.title = title
		self.plot_titles = plot_titles
		self.face_orientations = face_orientations
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
		orig_G = None 
		for obs in items:
			if isinstance(obs, GraphObservable):
				if orig_G is None:
					orig_G = obs.G
				else:
					assert obs.G is orig_G, 'Co-rendered observables must use the same graph'
		if orig_G != None:
			# TODO brittle
			orig_layout = self.layout_func(orig_G)
			G = nx.convert_node_labels_to_integers(orig_G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
			G = clear_attributes(G)
			layout = {i: orig_layout[n] for i, n in enumerate(orig_G.nodes())}
		def helper(obs: Observable, plot=None):
			# Domain-specific rendering
			if isinstance(obs, GraphObservable):
				if plot is None:
					plot = figure(x_range=self.x_rng, y_range=self.y_rng, tooltips=[], width=self.plot_width, height=self.plot_height) 
				plot.axis.visible = False
				plot.xgrid.grid_line_color = None
				plot.ygrid.grid_line_color = None
				renderer = from_networkx(G, layout)
				plot.renderers.append(renderer)
				plot.toolbar_location = None

				if obs.Gd is GraphDomain.nodes: 
					plot.add_tools(HoverTool(tooltips=[('value', '@value'), ('node', '@node')]))
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
						cbar.major_label_text_font_size = "15pt"
						plot.add_layout(cbar, 'right')
				elif obs.Gd is GraphDomain.edges:
					plot.add_tools(HoverTool(tooltips=[('value', '@value'), ('edge', '@edge')]))
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
						cbar.major_label_text_font_size = "15pt"
						plot.add_layout(cbar, 'right')
					if self.edge_colors:
						plot.renderers[0].edge_renderer.glyph = MultiLine(line_width=5, line_color=field('value', cmap))
					elif isinstance(obs, gds):
						plot.renderers[0].edge_renderer.data_source.data['thickness'] = [3 if (x in obs.X_dirichlet or x in obs.X_neumann) else 1 for x in obs.X] 
						plot.renderers[0].edge_renderer.glyph = MultiLine(line_width='thickness')
				elif obs.Gd is GraphDomain.faces:
					plot.add_tools(HoverTool(tooltips=[('value', '@value'), ('face', '@face')]))
					cmap = LinearColorMapper(palette=self.face_palette, low=self.face_rng[0], high=self.face_rng[1])
					self.face_cmaps[obs.plot_id] = cmap
					obs.face_source = ColumnDataSource()
					xs = [[orig_layout[n][0] for n in f] for f in obs.faces]
					ys = [[orig_layout[n][1] for n in f] for f in obs.faces]
					if hasattr(obs.G, 'rendered_faces'): # Hacky
						xs = [xs[i] for i in obs.G.rendered_faces]
						ys = [ys[i] for i in obs.G.rendered_faces]
					obs.face_source.data['xs'] = xs
					obs.face_source.data['ys'] = ys
					obs.face_source.data['value'] = np.zeros(len(xs))
					faces = Patches(xs='xs', ys='ys', fill_color=field('value', cmap), line_color='#FFFFFF', line_width=2)
					plot.add_glyph(obs.face_source, faces)
					if self.face_orientations:
						# TODO: only works for convex faces
						obs.centroid_x, obs.centroid_y = np.array([np.mean(row) for row in xs]), np.array([np.mean(row) for row in ys])
						obs.radius = 0.3 * np.array([min([np.sqrt((xs[i][j] - obs.centroid_x[i])**2 + (ys[i][j] - obs.centroid_y[i])**2) for j in range(len(xs[i]))]) for i in range(len(xs))])
						height = 2/5 * obs.radius
						arrows_ys = np.stack((obs.centroid_y-obs.radius, obs.centroid_y-obs.radius+height/2, obs.centroid_y-obs.radius-height/2), axis=1)
						obs.face_source.data['centroid_x'] = obs.centroid_x
						obs.face_source.data['centroid_y'] = obs.centroid_y
						obs.face_source.data['radius'] = obs.radius
						obs.face_source.data['arrows_ys'] = (arrows_ys + 0.01).tolist()
						self.draw_face_orientations(obs, cmap)
						arcs = Arc(x='centroid_x', y='centroid_y', radius='radius', start_angle=-0.9, end_angle=4.1, line_color=field('arrow_color', cmap))
						arrows = Patches(xs='arrows_xs', ys='arrows_ys', fill_color=field('arrow_color', cmap), line_color=field('arrow_color', cmap))
						plot.add_glyph(obs.face_source, arcs)
						plot.add_glyph(obs.face_source, arrows)
					if self.colorbars:
						cbar = ColorBar(color_mapper=cmap, ticker=BasicTicker(), title='face')
						cbar.major_label_text_font_size = "15pt"
						plot.add_layout(cbar, 'right')
				else:
					raise Exception('unknown graph domain.')
			elif isinstance(obs, PointObservable):
				plot = figure(width=self.plot_width, height=self.plot_height)
				plot.add_tools(HoverTool(tooltips=[('time', '@t'), ('value', '@value')]))
				plot.toolbar_location = None
				plot.x_range.follow = 'end'
				plot.x_range.follow_interval = 10.0
				plot.x_range.range_padding = 0
				plot.xaxis.major_label_text_font_size = "15pt"
				plot.xaxis.axis_label = 'Time'
				plot.yaxis.major_label_text_font_size = "15pt"
				plot.y_range.range_padding_units = 'absolute'
				plot.y_range.range_padding = obs.render_params['min_res'] / 2
				obs.src = ColumnDataSource({'t': [], 'value': []})
				# TODO: handle vector plotting
				glyph = Line(x='t', y='value')
				plot.add_glyph(obs.src, glyph)
				# plot.line('t', 'value', line_color='black', source=obs.src)
			else:
				raise Exception('unknown observable type: ', obs)
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

	def draw_face_orientations(self, obs, cmap):
		width = 3/5 * obs.radius
		handedness, absy = np.sign(obs.face_orientation_vector * obs.y), np.abs(obs.y)
		if hasattr(obs.G, 'rendered_faces'): # Hacky
			handedness = handedness[obs.G.rendered_faces]
			absy = absy[obs.G.rendered_faces]
		head, tail = obs.centroid_x + handedness * width / 2, obs.centroid_x - handedness * width / 2
		obs.face_source.data['arrows_xs'] = np.stack((head, tail, tail), axis=1).tolist()
		mid = (cmap.high + cmap.low) / 2
		parity = np.sign(absy - mid).clip(0)
		obs.face_source.data['arrow_color'] = parity * cmap.low + (1-parity) * cmap.high

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
				plot.title.text_font_size = '16pt'

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
				if isinstance(obs, GraphObservable):
					if obs.Gd is GraphDomain.nodes:
						self.plots[obs.plot_id].renderers[0].node_renderer.data_source.data['value'] = obs.y
						if self.dynamic_ranges:
							lo, hi = obs.y.min(), obs.y.max()
							mid = (lo+hi)/2
							lo, hi = min(lo, mid-self.min_rng_size/2) - self.range_padding, max(hi, mid+self.min_rng_size/2) + self.range_padding
							self.node_cmaps[obs.plot_id].low = lo
							self.node_cmaps[obs.plot_id].high = hi
					elif obs.Gd is GraphDomain.edges:
						self.draw_arrows(obs, obs.y)
						self.plots[obs.plot_id].renderers[0].edge_renderer.data_source.data['value'] = obs.arr_source.data['value']
						if self.dynamic_ranges:
							lo, hi = obs.arr_source.data['value'].min(), obs.arr_source.data['value'].max()
							mid = (lo+hi)/2
							lo, hi = min(lo, mid-self.min_rng_size/2) - self.range_padding, max(hi, mid+self.min_rng_size/2) + self.range_padding
							self.edge_cmaps[obs.plot_id].low = lo
							self.edge_cmaps[obs.plot_id].high = hi
					elif obs.Gd is GraphDomain.faces:
						absy = np.abs(obs.y)
						if hasattr(obs.G, 'rendered_faces'): # Hacky
							absy = absy[obs.G.rendered_faces]
						obs.face_source.data['value'] = absy
						if self.dynamic_ranges:
							lo, hi = absy.min(), absy.max()
							mid = (lo+hi)/2
							lo, hi = min(lo, mid-self.min_rng_size/2) - self.range_padding, max(hi, mid+self.min_rng_size/2) + self.range_padding
							self.face_cmaps[obs.plot_id].low = lo
							self.face_cmaps[obs.plot_id].high = hi
						if self.face_orientations:
							self.draw_face_orientations(obs, self.face_cmaps[obs.plot_id])
				elif isinstance(obs, PointObservable):
					# TODO: handle vector plotting
					obs.src.stream({'t': [obs.t], 'value': [obs.y]}, obs.render_params['retention'])
		if self.rec_name != None:
			self.dump_frame()

	def start_recording(self):
		self.rec_ctr = 0
		if not os.path.exists('recordings'):
			os.makedirs('recordings')
		self.rec_name = f'recordings/{now().strftime("%m-%d-%y %H:%M:%S")}'
		os.makedirs(self.rec_name)
		self.dump_frame()
		# chromedriver_path = str(Path(__file__).parent.parent.parent.parent / 'chromedriver')
		# assert os.path.exists(chromedriver_path), f'Cannot find chromedriver at: {chromedriver_path}'
		# self.webdriver = webdriver.Chrome(chromedriver_path)

	def dump_frame(self):
		# if hasattr(self, 'webdriver'):
			# export_png(self.root_plot, filename=f'{self.rec_name}/{self.rec_ctr}.png', timeout=10, webdriver=self.webdriver)
			# self.rec_ctr += 1
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
