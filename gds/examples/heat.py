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

def heat_test():
	G = gds.grid_graph(10, 10)
	temp1 = gds.node_gds(G)
	x = (5,5)
	i = temp1.X[x]
	B = temp1.incidence.copy().todense()
	B[i][B[i] < 0] = 0
	temp1.set_evolution(dydt=lambda t, y: -B@B.T@y)
	temp1.set_initial(y0=lambda n: 1.0 if n == x else 0.0)
	temp2 = gds.node_gds(G)
	temp2.set_evolution(dydt=lambda t, y: temp2.laplacian(y))
	temp2.set_constraints(dirichlet={x: 1.0})
	return temp1, temp2

if __name__ == '__main__':
	# Use coupling to visualize multiple PDEs simultaneously
	# p1 = grid_linear(steady_state=True)
	# p3 = grid_sinus_boundary(steady_state=True)	
	# p4 = grid_sinus_boundary(steady_state=True, phi=np.pi/4)

	# t1, t2 = heat_test()

	# sys = gds.couple({
		# 'heat0': p1,
		# 'heat1': t1,
		# 'heat2': t2
	# })

	# G = nx.random_geometric_graph(200, 0.125)
	# p5 = gds.node_gds(G)
	# p5.set_evolution(dydt=lambda t, y: 10*p5.laplacian())
	# p5.set_constraints({random.randint(0, 200): 1.0 for _ in range(10)})

	# gds.render(p5, title='Diffusion on a random geometric graph')
	# gds.render(sys, dynamic_ranges=True)

	G = gds.square_lattice(10, 10)
	temperature = gds.node_gds(G)
	temperature.set_evolution(dydt=lambda t, y: temperature.laplacian())
	temperature.set_constraints(dirichlet=gds.combine_bcs(
		lambda x: 0 if x[0] == 9 else None,
		lambda t, x: np.sin(t+x[1]/4)**2 if x[0] == 0 else None
	))
	gds.render(temperature, title='Heat equation with time-varying boundary')
