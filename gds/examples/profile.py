import networkx as nx
import numpy as np
import pdb
from itertools import count
import cProfile
import pstats

from gds.utils import set_seed
from gds import *

''' Definitions ''' 

def incompressible_ns_flow(G: nx.Graph, viscosity=1.0, density=1.0) -> (node_gds, edge_gds):
	velocity = edge_gds(G, dydt=lambda t, self: None)
	pressure = node_gds(G, 
		lhs=lambda t, self: self.gradient.T@velocity.advect_self() + self.laplacian()/density
	)
	velocity.dydt_fun = lambda t, self: -self.advect_self() - pressure.grad()/density + viscosity*self.laplacian()/density
	return pressure, velocity

def compressible_flow(G: nx.Graph, viscosity=1.0) -> (node_gds, edge_gds):
	pass

''' Systems ''' 

def poiseuille():
	m, n = 8, 10
	G = grid_graph(n, m)
	pressure, velocity = incompressible_ns_flow(G)
	def pressure_values(x):
		if x[0] == 0: return 0.2
		if x[0] == n-1: return -0.2
		return None
	pressure.set_constraints(neumann=pressure_values)
	def no_slip(x):
		if x[0][1] == x[1][1] == 0 or x[0][1] == x[1][1] == m-1:
			return 0.
		return None
	velocity.set_constraints(dirichlet=no_slip)
	return pressure, velocity

p, v = poiseuille()
sys = couple(p, v)

def run():
	global sys
	try:
		while True:
			sys[0].step(1e-2)
			print(sys[0].t)
	except KeyboardInterrupt:
		return

cProfile.run('run()', 'out.prof')
prof = pstats.Stats('out.prof')
prof.sort_stats('time').print_stats(40)
