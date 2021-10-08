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
from scipy.spatial.distance import pdist, squareform

import gds
from gds.types import *
from .fluid_projected import *

def run_experiment(foreach: Callable):
	eps = 1e-4
	steps = 10
	n_triangles = list(range(2, 7))
	energies = np.logspace(-1, 2, 5)

	fig, axs = plt.subplots(nrows=len(n_triangles), ncols=len(energies), figsize=(len(energies)*2, len(n_triangles)*2))

	for fig_i, N in enumerate(n_triangles):
		G = gds.triangular_lattice(m=1, n=N)
		N_e = len(G.edges())
		y0 = np.random.uniform(low=1, high=2, size=N_e)
		for fig_j, KE in enumerate(energies):
			V, P = euler(G)
			y0_ = V.leray_project(y0)
			y0_ *= np.sqrt(N_e * KE / np.dot(y0_, y0_))
			V.set_initial(y0=lambda e: y0_[V.X[e]])
			sys = gds.couple({'V': V, 'P': P})
			time, data = sys.solve(10, 0.01)
			foreach(time, data['V'], axs[fig_i][fig_j])
			if fig_i == 0:
				axs[fig_i][fig_j].set_title(f'{round(KE, 4)}')
			if fig_j == 0:
				axs[fig_i][fig_j].set_ylabel(f'{N}')

	fig.text(0.01, 0.5, '# Triangles', ha='center', va='center', rotation='vertical')
	fig.text(0.5, 0.99, 'Energy density (KE / |E|)', ha='center', va='center')
	plt.tight_layout(rect=[0.02, 0, 1, 0.98])
	plt.show()

def poincare_section():
	# Define system
	G = gds.triangular_lattice(m=1, n=2)
	N_e = len(G.edges())
	y0 = np.random.uniform(low=1, high=2, size=N_e)

	# Energies
	KE = np.linspace(1, 10, 10)
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
		time, data = sys.solve(20, 0.01)
		b = V.y[i]
		section = data['V'][np.round(data['V'] @ a, 2) == np.round(b, 2)]  
		axs[fig_idx].scatter(section[:, j], section[:, k], s=5)
		axs[fig_idx].set_title(f'KE: {ke}')

	plt.tight_layout()
	plt.show()

def recurrence_plot():
	eps = 1e-4
	steps = 10

	def foreach(time, data, ax):
		dists = pdist(data)
		dists = np.floor(dists/(eps*np.sqrt(data.shape[1])))
		dists[dists>steps] = steps
		points = squareform(dists)
		ax.imshow(points, origin='lower')

	run_experiment(foreach)

def stationarity():
	def foreach(time, data, ax):
		avg = np.cumsum(data, axis=0) / np.arange(1, data.shape[0]+1)[:,None]
		dist = np.linalg.norm(data - avg, axis=1)
		ax.plot(time, dist)

	run_experiment(foreach)


def velocity():
	eps = 1e-4
	steps = 10

	def foreach(time, data, ax):
		N_e = data.shape[1]
		heat = data[400:]
		heatmin, heatmax = heat.min(axis=0 , keepdims=True), heat.max(axis=0 , keepdims=True)
		heatmax += (eps*steps) / np.sqrt(N_e)
		heat -= heatmin
		heat /= heatmax
		ax.imshow(heat.T, aspect='auto')

	run_experiment(foreach)


if __name__ == '__main__':
	gds.set_seed(1)
	# poincare_section()
	# recurrence_plot()
	# stationarity()
	velocity()