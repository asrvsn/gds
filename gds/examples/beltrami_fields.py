'''
Computation of discrete Beltrami fields
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
from scipy.optimize import minimize, basinhopping
import cloudpickle

import gds
from gds.types import *
from .fluid_projected import *

n_triangles = list(range(2, 5))
n_ics = 10
save_path = 'runs/beltrami.pkl'

def solve_beltrami(G: nx.Graph):
	flow = gds.edge_gds(G)
	def f(x):
		return np.linalg.norm(flow.leray_project(flow.advect(flow.leray_project(x))))
	x0 = np.random.uniform(low=-10, high=10, size=flow.ndim)
	sol = basinhopping(f, x0)
	if not sol.success: 
		print('Unsuccessful solve')
		pdb.set_trace()
	u = flow.leray_project(sol.x)
	flow.set_evolution(nil=True)
	flow.set_initial(y0=lambda x: u[flow.X[x]])
	return flow

def plot_beltrami():
	sys = dict()
	for N in n_triangles:
		for i in range(n_ics):
			print((N, i))
			G = gds.triangular_lattice(m=1, n=N)
			sys[f'{N}_{i}'] = solve_beltrami(G)
	sys = gds.couple(sys)
	canvas = gds.grid_canvas(sys.observables.values(), n_ics)
	gds.render(sys, canvas=canvas, edge_max=0.6, dynamic_ranges=True)

def save_beltrami()
	data = dict()
	for N in n_triangles:
		data[N] = dict()
		for i in range(n_ics):
			print((N, i))
			G = gds.triangular_lattice(m=1, n=N)
			flow = solve_beltrami(G)
			data[N][i] = flow.y
	with open(save_path, 'wb') as f:
		cloudpickle.dump(data, f)

def analyze_beltrami(foreach):
	if not os.path.exists(save_path):
		raise Exception('no data')

	with open(save_path, 'rb') as f:
		data = cloudpickle.load(f)
		fig_N, fig_M = len(data), len(data[0])
		fig, axs = plt.subplots(nrows=fig_N, ncols=fig_M, figsize=(fig_M*2, fig_N*2))
		for fig_i, (N, subdata) in enumerate(sorted(data.items(), lambda x: x[0])):
			for fig_j, i in enumerate(subdata):
				foreach(N, subdata[i], axs[fig_i][fig_j])
				if fig_i < fig_N-1:
					axs[fig_i][fig_j].axes.xaxis.set_visible(False)
				if fig_j == 0:
					axs[fig_i][fig_j].set_ylabel(f'{N}')
				else:
					axs[fig_i][fig_j].axes.yaxis.set_visible(False)
		fig.text(0.01, 0.5, '# Triangles', ha='center', va='center', rotation='vertical')
		plt.tight_layout(rect=[0.02, 0, 1, 0.98])
		plt.show()


if __name__ == '__main__':
	gds.set_seed(1)

	plot_beltrami()
	save_beltrami()
	

