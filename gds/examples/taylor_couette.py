import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
from .fluid import *

def taylor_couette():
	G = gds.flat_prism(k=20, n=20, inner_radius=5)
	inner = G.subgraph(range(20)).edges()
	outer = G.subgraph(range(380, 400)).edges()
	outer_v = 100.0
	inner_v = 3.0
	velocity, pressure = navier_stokes(G, viscosity=10)
	def rotation(x):
		if x in inner:
			return 0
		# 	return -inner_v if 19 in x else inner_v
		if x in outer:
			return -outer_v if 399 in x else outer_v
	velocity.set_constraints(dirichlet=rotation)
	return velocity, pressure


if __name__ == '__main__':
	v, p = taylor_couette()
	fluid_test(v, p)