'''
Shear layers
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
from .fluid_projected import *

''' Systems ''' 

def double_periodic_shear():
	M, N = 20, 20
	G = gds.square_lattice(M, N, periodic=True)
	G.faces, _ = gds.embedded_faces(gds.square_lattice(M, N)) # Ignore periodic edges in faces
	velocity, pressure = euler(G)
	eps = 0.1
	def y0(e):
		(n1, n2) = e
		(x1, y1) = n1
		(x2, y2) = n2
		if y1 == y2:
			sign = 1 if abs(x1 - x2) == 1 else -1
			return sign * (np.tanh(y1/M-0.25) if y1/M <= 0.5 else np.tanh(0.75-y1/M))
		elif x1 == x2:
			sign = 1 if abs(y1 - y2) == 1 else -1
			return sign * eps * np.sin(2*np.pi*x1/M)
	velocity.set_initial(y0=y0)
	return velocity, pressure


if __name__ == '__main__':

	v, p = double_periodic_shear()
	fluid_test(v, p, columns=3, edge_palette=cc.bgy)