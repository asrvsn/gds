import networkx as nx
import numpy as np
import pdb
from itertools import count
import cProfile
import pstats

from gpde.utils import set_seed
from gpde import *

''' Definitions ''' 

def incompressible_flow(G: nx.Graph, viscosity=1.0, density=1.0) -> (vertex_pde, edge_pde):
	velocity = edge_pde(G, dydt=lambda t, self: None)
	pressure = vertex_pde(G, 
		lhs=lambda t, self: self.gradient.T@velocity.advect_self() + self.laplacian()/density
	)
	velocity.dydt_fun = lambda t, self: -self.advect_self() - pressure.grad()/density + viscosity*self.laplacian()/density
	return pressure, velocity

def compressible_flow(G: nx.Graph, viscosity=1.0) -> (vertex_pde, edge_pde):
	pass

''' Systems ''' 

def poiseuille():
	m, n = 8, 10
	G = nx.grid_2d_graph(n, m)
	pressure, velocity = incompressible_flow(G)
	def pressure_values(t, x):
		if x[0] == 0: return 0.2
		if x[0] == n-1: return -0.2
		return None
	pressure.set_boundary(neumann=pressure_values, dynamic=False)
	def no_slip(t, x):
		if x[0][1] == x[1][1] == 0 or x[0][1] == x[1][1] == m-1:
			return 0.
		return None
	velocity.set_boundary(dirichlet=no_slip, dynamic=False)
	return pressure, velocity

def run():
	try:
		p, v = poiseuille()
		sys = couple(p, v)
		while True:
			sys[0].step(1e-2)
			print(sys[0].t)
	except KeyboardInterrupt:
		return

cProfile.run('run()', 'out.prof')
prof = pstats.Stats('out.prof')
prof.sort_stats('time').print_stats(40)
