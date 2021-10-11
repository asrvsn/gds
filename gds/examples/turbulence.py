'''
Turbulence analysis of incompressible fluids.
'''

import networkx as nx
import numpy as np
import pdb
import os
import shutil
from itertools import count
import colorcet as cc
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.signal import stft
from scipy.fft import rfftn
import statsmodels.api as sm

import gds
from gds.types import *
from .fluid_projected import *

folder = 'runs/turbulence'

def solve(T=20, dt=0.01):
	if os.path.isdir(folder):
		shutil.rmtree(folder)
	os.mkdir(folder)

	n_triangles = list(range(2, 7))
	energies = np.logspace(-1, 1.5, 5)

	for N in n_triangles:
		os.mkdir(f'{folder}/{N}')

		G = gds.triangular_lattice(m=1, n=N)
		N_e = len(G.edges())
		y0 = np.random.uniform(low=1, high=2, size=N_e)

		for KE in energies:
			V, P = euler(G)
			y0_ = V.leray_project(y0)
			y0_ *= np.sqrt(N_e * KE / np.dot(y0_, y0_))
			V.set_initial(y0=lambda e: y0_[V.X[e]])
			sys = gds.couple({'V': V, 'P': P})
			time, data = sys.solve(T, dt)

			with open(f'{folder}/{N}/{KE}.npy', 'wb') as f:
				np.save(f, data['V'])
			

def analyze(foreach: Callable):
	if not os.path.isdir(folder):
		raise Exception('no data')

	n_triangles = [int(s) for s in os.listdir(folder)]
	energies = [float(s[:-4]) for s in os.listdir(f'{folder}/{n_triangles[0]}')]

	fig, axs = plt.subplots(nrows=len(n_triangles), ncols=len(energies), figsize=(len(energies)*2, len(n_triangles)*2))

	for fig_i, N in enumerate(sorted(n_triangles)):
		for fig_j, KE in enumerate(sorted(energies)):
			with open(f'{folder}/{N}/{KE}.npy', 'rb') as f:
				data = np.load(f)
				foreach(data, axs[fig_i][fig_j])
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

def recurrence():
	eps = 1e-3
	steps = 10

	def foreach(data, ax):
		dists = pdist(data[1000:2000])
		dists = np.floor(dists/(eps*np.sqrt(data.shape[1])))
		dists[dists>steps] = steps
		points = squareform(dists)
		ax.imshow(points, origin='lower')

	analyze(foreach)

def stationarity():
	def foreach(data, ax):
		avg = np.cumsum(data, axis=0) / np.arange(1, data.shape[0]+1)[:,None]
		dist = np.linalg.norm(data - avg, axis=1)
		ax.plot(dist)

	analyze(foreach)


def velocity():
	eps = 1e-4
	steps = 10

	def foreach(data, ax):
		N_e = data.shape[1]
		heat = data[400:]
		# heatmin, heatmax = heat.min(axis=0 , keepdims=True), heat.max(axis=0 , keepdims=True)
		# heatmax += (eps*steps) / np.sqrt(N_e)
		# heat -= heatmin
		# heat /= heatmax
		sns.heatmap(heat.T, ax=ax, norm=LogNorm())

	analyze(foreach)

def temporal_fourier_transform():

	def foreach(data, ax):
		fs = np.abs(rfftn(data[400:].T))
		sns.heatmap(fs, ax=ax, norm=LogNorm())

	analyze(foreach)

def spatial_fourier_transform():

	def foreach(data, ax):
		fs = np.abs(rfftn(data[400:].T))
		sns.heatmap(fs, ax=ax, cbar=False)

	analyze(foreach)

def autocorrelation():

	def foreach(data, ax):
		data = data[400:].T
		corr = []
		for series in data:
			corr.append(sm.tsa.acf(series, nlags=100, fft=True))
		corr = np.array(corr)
		sns.heatmap(corr, ax=ax)

	analyze(foreach)

if __name__ == '__main__':
	gds.set_seed(1)
	# solve()
	# poincare_section()
	recurrence()
	# stationarity()
	# velocity()
	# temporal_fourier_transform()
	# autocorrelation()