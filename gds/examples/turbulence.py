'''
Turbulence analysis of incompressible fluids.
'''

import networkx as nx
import numpy as np
import pdb
from itertools import count
import colorcet as cc
import random
import matplotlib.pyplot as plt

import gds
from gds.types import *
from .fluid_projected import *

def poincare_section():
	# Define system
	G = gds.triangular_lattice(m=1, n=5)
	N_e = len(G.edges())
	y0 = np.random.uniform(low=1, high=2, size=N_e)

	# Energies
	KE = np.linspace(1, 15, 5)
	fig, axs = plt.subplots(nrows=1, ncols=len(KE), figsize=(len(KE)*5, 5))

	# Define transverse hyperplane
	i = random.randint(0, N_e)
	a = np.zeros(N_e)
	a[i] = 1
	j, k = random.randint(0, N_e), random.randint(0, N_e)
	while j == i or k == i:
		j, k = random.randint(0, N_e), random.randint(0, N_e)

	# Solve systems & Plot SOS
	for fig_idx, ke in enumerate(KE):
		print(ke)
		V, P = euler(G)
		y0_ = V.leray_project(y0)
		y0_ *= np.sqrt(ke / np.dot(y0_, y0_))
		V.set_initial(y0=lambda e: y0_[V.X[e]])
		sys = gds.couple({'V': V, 'P': P})
		time, data = sys.solve(10, 0.01)
		b = V.y[i]
		section = data['V'][np.round(data['V'] @ a, 2) == np.round(b, 2)]  
		axs[fig_idx].scatter(section[:, j], section[:, k], s=5**2)
		axs[fig_idx].set_title(f'KE: {ke}')

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	gds.set_seed(10)
	poincare_section()