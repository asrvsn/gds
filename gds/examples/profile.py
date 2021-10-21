import networkx as nx
import numpy as np
import pdb
from itertools import count
import cProfile
import pstats

import gds
from gds.utils import set_seed
import gds.examples.fluid as fluid
import gds.examples.fluid_projected as fluid_projected
from gds.examples.von_karman import von_karman_projected

''' Systems ''' 

def poiseuille():
	m, n = 8, 10
	G = gds.square_lattice(n, m)
	velocity, pressure = fluid_projected.navier_stokes(G)
	def pressure_values(x):
		if x[0] == 0: return 0.2
		if x[0] == n-1: return -0.2
		return None
	pressure.set_constraints(dirichlet=pressure_values)
	def no_slip(x):
		if x[0][1] == x[1][1] == 0 or x[0][1] == x[1][1] == m-1:
			return 0.
		return None
	velocity.set_constraints(dirichlet=no_slip)
	return velocity, pressure

if __name__ == '__main__':
	gds.set_seed(1)
	# v, p = poiseuille()
	# v, p = von_karman_projected()
	G = gds.triangular_lattice(m=1, n=3)
	v, p = fluid_projected.random_euler(G, 10)
	sys = gds.couple({'v': v, 'p': p})

	def run():
		global sys
		try:
			while True:
				sys.step(1e-2)
				print(sys.t)
		except KeyboardInterrupt:
			return

	cProfile.run('run()', 'out.prof')
	prof = pstats.Stats('out.prof')
	prof.sort_stats('time').print_stats(40)
