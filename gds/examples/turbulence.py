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
			y0_ *= np.sqrt(KE / np.dot(y0_, y0_))
			V.set_initial(y0=lambda e: y0_[V.X[e]])
			sys = gds.couple({'V': V, 'P': P})
			time, data = sys.solve(10, 0.01)
			dists = pdist(data['V'])
			dists = np.floor(dists/(eps*np.sqrt(y0_.size)))
			dists[dists>steps] = steps
			points = squareform(dists)
			axs[fig_i][fig_j].imshow(points, origin='lower')
			if fig_i == 0:
				axs[fig_i][fig_j].set_title(f'KE: {KE}')
			if fig_j == 0:
				axs[fig_i][fig_j].set_ylabel(f'N: {N}')

	plt.tight_layout()
	plt.show()

def stationarity():
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
			y0_ *= np.sqrt(KE / np.dot(y0_, y0_))
			V.set_initial(y0=lambda e: y0_[V.X[e]])
			sys = gds.couple({'V': V, 'P': P})
			time, data = sys.solve(20, 0.01)
			avg = np.cumsum(data['V'], axis=0) / np.arange(1, data['V'].shape[0]+1)[:,None]
			dist = np.linalg.norm(data['V'] - avg, axis=1)
			axs[fig_i][fig_j].plot(time, dist)
			if fig_i == 0:
				axs[fig_i][fig_j].set_title(f'KE: {KE}')
			if fig_j == 0:
				axs[fig_i][fig_j].set_ylabel(f'N: {N}')

	plt.tight_layout()
	plt.show()

def velocity():
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
			y0_ *= np.sqrt(KE / np.dot(y0_, y0_))
			V.set_initial(y0=lambda e: y0_[V.X[e]])
			sys = gds.couple({'V': V, 'P': P})
			time, data = sys.solve(10, 0.01)
			heat = np.abs(data['V'][400:])
			heat -= heat.min(axis=0 , keepdims=True)
			heat /= heat.max(axis=0 , keepdims=True)
			# pdb.set_trace()
			axs[fig_i][fig_j].imshow(heat.T, aspect='auto')
			if fig_i == 0:
				axs[fig_i][fig_j].set_title(f'KE: {KE}')
			if fig_j == 0:
				axs[fig_i][fig_j].set_ylabel(f'N: {N}')

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	gds.set_seed(1)
	# poincare_section()
	# recurrence_plot()
	# stationarity()
	velocity()