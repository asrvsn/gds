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
from numpy.fft import rfftn, fft
from scipy.stats import beta, ortho_group
import statsmodels.api as sm
from pyunicorn.timeseries import RecurrencePlot
import torch

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
start, end = None, None

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
			

def analyze_each(foreach: Callable):
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
				foreach(G, data[start:end], axs[fig_i][fig_j])
				if fig_i == 0:
					axs[fig_i][fig_j].set_title(f'{round(KE, 4)}')
				if fig_i < fig_N-1:
					axs[fig_i][fig_j].axes.xaxis.set_visible(False)
				if fig_j == 0:
					axs[fig_i][fig_j].set_ylabel(f'{N}')
				# else:
				# 	axs[fig_i][fig_j].axes.yaxis.set_visible(False)

	fig.text(0.01, 0.5, '# Triangles', ha='center', va='center', rotation='vertical')
	fig.text(0.5, 0.99, 'Energy density (KE / |E|)', ha='center', va='center')
	plt.tight_layout(rect=[0.02, 0, 1, 0.98])
	plt.show()

def analyze_all(foreach: Callable):
	if not os.path.isdir(folder):
		raise Exception('no data')

	n_triangles = [int(s) for s in os.listdir(folder)]
	energies = [float(s[:-4]) for s in os.listdir(f'{folder}/{n_triangles[0]}')]

	for fig_i, N in enumerate(sorted(n_triangles)):
		for fig_j, KE in enumerate(sorted(energies)):
			print((fig_i, fig_j))
			G = gds.triangular_lattice(m=1, n=N)
			with open(f'{folder}/{N}/{KE}.npy', 'rb') as f:
				data = np.load(f)
				foreach(G, data[start:end], N, KE)

def poincare_section():
	indices = dict()
	def foreach(G, data, ax):
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

	analyze_each(foreach)


def recurrence():
	eps = 1e-4
	steps = 10

	def foreach(G, data, ax):
		idx = 0
		rp = RecurrencePlot(data[:,idx], threshold=0.3, metric='supremum', normalize=True)
		mat = rp.recurrence_matrix()
		ax.imshow(mat, origin='lower')
		if fig_i != fig_M-1:
			ax.axes.xaxis.set_visible(False)
		if fig_j != 0:
			ax.axes.yaxis.set_visible(False)

	analyze_each(foreach)

def stationarity():
	def foreach(G, data, ax):
		avg = np.cumsum(data, axis=0) / np.arange(1, data.shape[0]+1)[:,None]
		dist = np.linalg.norm(data - avg, axis=1)
		dist = np.round(dist, 8) 
		ax.plot(dist)

	analyze_each(foreach)


def turbulence_kinetic_energy():
	def foreach(G, data, ax):
		avg = rolling_mean_2d(data)
		fluct = data - avg
		fluct_var = rolling_mean_2d(fluct ** 2)
		tke = fluct_var.sum(axis=1) / data.shape[1]
		tke = np.round(tke, 15) # Truncate to machine precision (1e-15)
		ax.plot(tke)

	analyze_each(foreach)


def raw_data():

	def foreach(G, data, ax):
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

	analyze_each(foreach)

def temporal_fourier_transform():

	def foreach(G, data, ax):
		fs = np.abs(rfftn(data.T))
		sns.heatmap(fs, ax=ax, norm=LogNorm())

	analyze_each(foreach)

def power_spectrum():

	def foreach(G, data, ax):
		freqs, spec_fun = edge_power_spectrum(G, method='hodge_cycles')
		spectrum = spec_fun(data.T)
		# spectrum = (freqs_ ** (-5/3))[:,np.newaxis] # Theoretical energy distribution
		sns.heatmap(spectrum, ax=ax, cbar=False, yticklabels=freqs) 
		ax.invert_yaxis() # (reversed order for sns)

	analyze_each(foreach)

def power_spectrum_fft():

	def foreach(G, data, ax):
		freqs, spec_fun = edge_power_spectrum(G, method='hodge_cycles')
		spectrum = spec_fun(data.T)
		# spectrum = (freqs_ ** (-5/3))[:,np.newaxis] # Theoretical energy distribution
		spectrum = np.abs(fft(spectrum, axis=1))
		sns.heatmap(spectrum, ax=ax, yticklabels=freqs, norm=LogNorm()) 
		ax.invert_yaxis() # (reversed order for sns)

	analyze_each(foreach)

def autocorrelation():

	def foreach(G, data, ax):
		data = data.T
		corr = []
		for series in data:
			corr.append(sm.tsa.acf(series, nlags=100, fft=True))
		corr = np.array(corr)
		sns.heatmap(corr, ax=ax)

	analyze_each(foreach)

def energy_drift():

	def foreach(G, data, ax):
		data = np.linalg.norm(data, axis=1)
		ax.plot(data)

	analyze_each(foreach)

def dKdt():

	def foreach(G, data, ax):
		v = gds.edge_gds(G)
		values = []
		for row in data:
			values.append(np.dot(row, v.leray_project(-v.advect(row))))
		ax.plot(values)

	analyze_each(foreach)

def lyapunov_spectra():
	transient = 2000
	window = 1000
	interval = 10
	floor = -10

	plot_data = dict()

	def foreach(G, data, N, KE):
		v = gds.edge_gds(G)
		P = v.leray_projector
		data = torch.from_numpy(data).float()
		spectra = 0
		du_t = ortho_group.rvs(data.shape[1]) # Initial orthonormal perturbation matrix
		for i in range(window):
			u_t = data[transient + i]
			J_t = P @ v.advect_jac(u_t).numpy()
			du_t += dt * J_t @ du_t
			if i % interval == 0:
				Q, R = np.linalg.qr(du_t, mode='complete')
				spectra += np.log(np.abs(np.diag(R)))
				du_t = Q
		spectra /= (window / interval)
		spectra = spectra[spectra >= floor]
		if not (N in plot_data):
			plot_data[N] = {'x': [], 'y': [], 'e_x': [], 'e_y': []}
		plot_data[N]['x'].extend([KE] * spectra.size)
		plot_data[N]['y'].extend(spectra.tolist())
		plot_data[N]['e_x'].append(KE)
		plot_data[N]['e_y'].append(spectra.max())

	analyze_all(foreach)

	m = len(plot_data)
	fig, axs = plt.subplots(nrows=m, ncols=1, figsize=(m*2, m*2))
	for fig_i, N in enumerate(plot_data):
		axs[fig_i].scatter(plot_data[N]['x'], plot_data[N]['y'], s=1, color='blue')
		axs[fig_i].plot(plot_data[N]['e_x'], plot_data[N]['e_y'], color='black', alpha=0.5)
		axs[fig_i].plot(plot_data[N]['e_x'], [0] * len(plot_data[N]['e_x']), color='black')
		axs[fig_i].set_ylabel(f'{N}')
		axs[fig_i].set_xscale('log')
		if fig_i < m-1:
			axs[fig_i].axes.xaxis.set_visible(False)
	fig.text(0.01, 0.5, '# Triangles', ha='center', va='center', rotation='vertical')
	fig.text(0.5, 0.01, 'Energy density (KE / |E|)', ha='center', va='center')
	plt.tight_layout(rect=[0.02, 0.02, 1, 1])
	plt.show()

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
	# power_spectrum()
	# power_spectrum_fft()
	# autocorrelation()
	# energy_drift()
	# dKdt()
	lyapunov_spectra()