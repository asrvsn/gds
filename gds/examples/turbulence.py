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
from scipy.stats import beta
import statsmodels.api as sm
from pyunicorn.timeseries import RecurrencePlot

import gds
from gds.types import *
from gds.utils import rolling_mean_2d
from .fluid import initial_flow, edge_power_spectrum
from .fluid_projected import *

folder = 'runs/turbulence_flat_energy'
n_triangles = list(range(2, 7))
energies = np.logspace(-1, 2, 10)
# n_triangles = list(range(2, 4))
# energies = np.logspace(-1, 1.5, 5)
fig_N, fig_M = len(n_triangles), len(energies)
T = 80
dt = 0.01

beta_rv = beta(5, 1)
# scale_distribution = lambda x: 1.0 # Uniform energy distribution across length scales
# scale_distribution = lambda x: beta_rv.pdf(x) # High-energy distribution across length scales
scale_distribution = lambda x: beta_rv.pdf(1-x) # Low-energy distribution across length scales
start, end = -2000, None

def solve():
	if os.path.isdir(folder):
		shutil.rmtree(folder)
	os.mkdir(folder)

	for N in n_triangles:
		os.mkdir(f'{folder}/{N}')

		G = gds.triangular_lattice(m=1, n=N)
		N_e = len(G.edges())
		y0 = initial_flow(G, scale_distribution=scale_distribution)

		for KE in energies:
			V, P = euler(G)
			y0_ = y0 * np.sqrt(N_e * KE / np.dot(y0, y0))
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
			print((fig_i, fig_j))
			G = gds.triangular_lattice(m=1, n=N)
			with open(f'{folder}/{N}/{KE}.npy', 'rb') as f:
				data = np.load(f)
				foreach(G, data[start:end], axs[fig_i][fig_j], fig_i, fig_j)
				if fig_i == 0:
					axs[fig_i][fig_j].set_title(f'{round(KE, 4)}')
				if fig_i < fig_N-1:
					axs[fig_i][fig_j].axes.xaxis.set_visible(False)
				if fig_j == 0:
					axs[fig_i][fig_j].set_ylabel(f'{N}')
				else:
					axs[fig_i][fig_j].axes.yaxis.set_visible(False)

	fig.text(0.01, 0.5, '# Triangles', ha='center', va='center', rotation='vertical')
	fig.text(0.5, 0.99, 'Energy density (KE / |E|)', ha='center', va='center')
	plt.tight_layout(rect=[0.02, 0, 1, 0.98])
	plt.show()

def poincare_section():
	indices = dict()
	def foreach(G, data, ax, fig_i, fig_j):
		# Define transverse hyperplane
		M = data.shape[1]
		if not M in indices:
			i = random.randint(0, M)
			j, k = i, i
			while j == i or k == i:
				j, k = random.randint(0, M-1), random.randint(0, M-1)
			indices[M] = (i, j, k)
		(i, j, k) = indices[M]
		a = np.zeros(M)
		a[i] = 1
		b = data[-1, i]
		section = data[np.abs(data@a - b) <= 5e-1]
		ax.scatter(section[:, j], section[:, k], s=1)

	analyze(foreach)


def recurrence():
	eps = 1e-4
	steps = 10

	def foreach(G, data, ax, fig_i, fig_j):
		idx = 0
		rp = RecurrencePlot(data[:,idx], threshold=0.3, metric='supremum', normalize=True)
		mat = rp.recurrence_matrix()
		ax.imshow(mat, origin='lower')
		if fig_i != fig_M-1:
			ax.axes.xaxis.set_visible(False)
		if fig_j != 0:
			ax.axes.yaxis.set_visible(False)

	analyze(foreach)

def stationarity():
	def foreach(G, data, ax, fig_i, fig_j):
		avg = np.cumsum(data, axis=0) / np.arange(1, data.shape[0]+1)[:,None]
		dist = np.linalg.norm(data - avg, axis=1)
		ax.plot(dist)

	analyze(foreach)


def turbulence_kinetic_energy():
	def foreach(G, data, ax, fig_i, fig_j):
		avg = rolling_mean_2d(data)
		fluct = data - avg
		fluct_var = rolling_mean_2d(fluct ** 2)
		tke = fluct_var.sum(axis=1) / data.shape[1]
		tke = np.round(tke, 15) # Truncate to machine precision (1e-15)
		ax.plot(tke)

	analyze(foreach)


def raw_data():

	def foreach(G, data, ax, fig_i, fig_j):
		# N_e = data.shape[1]
		heat = data
		heatmin, heatmax = heat.min(axis=0 , keepdims=True), heat.max(axis=0 , keepdims=True)
		# heatmax += (eps*steps) / np.sqrt(N_e)
		heat -= heatmin
		# pdb.set_trace()
		delta = heatmax - heatmin
		delta[delta == 0] = 1
		heat /= delta
		sns.heatmap(heat.T, ax=ax, cbar=False)
		ax.invert_yaxis() # (reversed order for sns)

	analyze(foreach)

def temporal_fourier_transform():

	def foreach(G, data, ax, fig_i, fig_j):
		fs = np.abs(rfftn(data.T))
		sns.heatmap(fs, ax=ax, norm=LogNorm())

	analyze(foreach)

def power_spectrum():

	def foreach(G, data, ax, fig_i, fig_j):
		freqs, spec_fun = edge_power_spectrum(G, method='hodge_cycles')
		if fig_i == 3:
			pdb.set_trace()
		spectrum = spec_fun(data.T)
		# spectrum = (freqs_ ** (-5/3))[:,np.newaxis] # Theoretical energy distribution
		sns.heatmap(spectrum, ax=ax, cbar=False, yticklabels=freqs) 
		ax.invert_yaxis() # (reversed order for sns)

	analyze(foreach)

def autocorrelation():

	def foreach(G, data, ax, fig_i, fig_j):
		data = data.T
		corr = []
		for series in data:
			corr.append(sm.tsa.acf(series, nlags=100, fft=True))
		corr = np.array(corr)
		sns.heatmap(corr, ax=ax)

	analyze(foreach)

def energy_drift():

	def foreach(G, data, ax, fig_i, fig_j):
		data = np.linalg.norm(data, axis=1)
		ax.plot(data)

	analyze(foreach)

def dKdt():

	def foreach(G, data, ax, fig_i, fig_j):
		v = gds.edge_gds(G)
		values = []
		for row in data:
			values.append(np.dot(row, v.leray_project(-v.advect(row))))
		ax.plot(values)

	analyze(foreach)

if __name__ == '__main__':
	gds.set_seed(1)
	# plt.plot(np.linspace(0,1,10), [scale_distribution(x) for x in np.linspace(0,1,10)])
	# plt.show()
	# solve()
	# poincare_section()
	# recurrence()
	# turbulence_kinetic_energy()
	# stationarity()
	# raw_data()
	# temporal_fourier_transform()
	power_spectrum()
	# autocorrelation()
	# energy_drift()
	# dKdt()