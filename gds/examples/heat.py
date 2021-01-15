''' Heat equation ''' 

import numpy as np
import networkx as nx
import pdb

from gds import *
from gds.render.bokeh import *
from gds.utils.graph import *

def heat_grid(n = 10, steady_state=False) -> node_gds:
	G = grid_graph(n, n)
	temp = node_gds(G)
	if steady_state:
		''' Steady-state version (for comparison) ''' 
		temp.set_evolution(cost=lambda t, y: temp.laplacian(y))
	else:
		temp.set_evolution(dydt=lambda t, y: temp.laplacian(y))
	return temp

def grid_const_boundary(steady_state=False) -> node_gds:
	n = 7
	temp = heat_grid(n=n, steady_state=steady_state)
	temp.set_constraints(dirichlet = lambda x: 1.0 if (0 in x or (n-1) in x) else None)
	return temp

def grid_mixed_boundary(steady_state=False) -> node_gds:
	n = 10
	temp = heat_grid(n=n, steady_state=steady_state)
	def dirichlet(t, x):
		if x[0] == 0 or x[0] == n-1:
			return 0.5
		elif x[1] == 0:
			return 1.0
		return None
	def neumann(t, x):
		if x[1] == n-1 and x[0] not in (0, n-1):
			return -0.1
		return None
	temp.set_constraints(dirichlet=dirichlet, neumann=neumann)
	return temp

def grid_timevarying_boundary(steady_state=False) -> node_gds:
	n = 10
	temp = heat_grid(n=n, steady_state=steady_state)
	temp.set_constraints(
		dirichlet = lambda t, x: np.sin(t/5)**2 if (0 in x or (n-1) in x) else None
	)
	return temp

def grid_linear(steady_state=False) -> node_gds:
	n = 10
	temp = heat_grid(n=n, steady_state=steady_state)
	def dirichlet(x):
		if x[0] == 0:
			return 1.0
		elif x[0] == n-1:
			return 0.
		return None
	temp.set_constraints(dirichlet=dirichlet)
	return temp

if __name__ == '__main__':
	# Use coupling to visualize multiple PDEs simultaneously
	p1 = grid_linear()
	p2 = grid_linear(steady_state=True)
	# sys = couple({
	# 	'heat1': p1,
	# 	'heat2': p2
	# })
	sys = p2.system('heat')
	LiveRenderer(sys, grid_canvas([p2])).start()
