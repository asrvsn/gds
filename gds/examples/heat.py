''' Heat equation ''' 

import numpy as np
import networkx as nx
import pdb
import random

import gds

def heat_grid(n = 10, steady_state=False) -> gds.node_gds:
	G = gds.grid_graph(n, n)
	temp = gds.node_gds(G)
	if steady_state:
		''' Steady-state version (for comparison) ''' 
		temp.set_evolution(cost=lambda t, y: temp.laplacian(y))
	else:
		temp.set_evolution(dydt=lambda t, y: temp.laplacian(y))
	return temp

def grid_const_boundary(steady_state=False) -> gds.node_gds:
	n = 7
	temp = heat_grid(n=n, steady_state=steady_state)
	temp.set_constraints(dirichlet = lambda x: 1.0 if (0 in x or (n-1) in x) else None)
	return temp

def grid_mixed_boundary(steady_state=False) -> gds.node_gds:
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

def grid_sinus_boundary(steady_state=False, phi=0) -> gds.node_gds:
	n = 10
	temp = heat_grid(n=n, steady_state=steady_state)
	def dirichlet(t, x):
		if x[0] == 0:
			return np.sin(t+x[1]/4+phi)**2
		elif x[0] == n-1:
			return 0.
		return None
	temp.set_constraints(dirichlet=dirichlet)
	return temp

def grid_linear(steady_state=False) -> gds.node_gds:
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
	p1 = grid_linear(steady_state=True)
	p3 = grid_sinus_boundary(steady_state=True)	
	p4 = grid_sinus_boundary(steady_state=True, phi=np.pi/4)

	sys = gds.couple({
		# 'heat0': p1,
		'heat1': p3,
		'heat2': p4
	})

	G = nx.random_geometric_graph(200, 0.125)
	p5 = gds.node_gds(G)
	p5.set_evolution(dydt=lambda t, y: 10*p5.laplacian())
	p5.set_constraints({random.randint(0, 200): 1.0 for _ in range(10)})

	gds.render(p5, title='Diffusion on a random geometric graph')
