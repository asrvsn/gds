''' 
Quantum systems 
units h = m = 1 unless otherwise set
''' 

import numpy as np
import networkx as nx
import pdb
import colorcet as cc
from typing import Callable

import gds

def schrodinger(G: nx.Graph, V: Callable, psi0: Callable, **kwargs):
	re, im = gds.node_gds(G), gds.node_gds(G)
	V_arr = np.array([V(x) for x in re.X])
	def f(t, y):
		psi = re.y + 1j*im.y
		return 1j * (re.laplacian(psi) - V_arr*psi)
	re.set_evolution(dydt=lambda t, y: np.real(f(t, y)))
	re.set_initial(y0=lambda x: np.real(psi0(x)))
	im.set_evolution(dydt=lambda t, y: np.imag(f(t, y)))
	im.set_initial(y0=lambda x: np.imag(psi0(x)))
	sys = gds.couple({
		'real': re,
		'imag': im,
	})
	gds.render(sys, dynamic_ranges=True, node_palette=cc.bmy, n_spring_iters=1000, **kwargs)

def toroidal_harmonic_oscillator():
	G = gds.torus()
	V = lambda x: x[0] + x[1]
	psi0 = lambda x: x[0] + 1j*x[1]
	schrodinger(G, V, psi0, title='Quantum harmonic oscillator on a torus')

if __name__ == '__main__':
	# wave()
	toroidal_harmonic_oscillator()
	