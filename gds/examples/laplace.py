''' Laplace equation, or steady-state heat heat transfer ''' 

import pdb
import gds
import networkx as nx
import numpy as np

def sinus_boundary():
	n = 10
	G = grid_graph(n, n)
	eq = node_gds(G, lhs=lambda t, self: self.laplacian())
	def dirichlet(x):
		if x[0] == 0:
			return np.sin(x[1]/10)
		return None
	eq.set_constraints(dirichlet=dirichlet)
	return eq

def sinus_boundary_timevarying():
	n = 10
	G = grid_graph(n, n)
	eq = node_gds(G, lhs=lambda t, self: self.laplacian(), gtol=1e-8)
	def dirichlet(t, x):
		if x[0] == 0:
			return np.sin(x[1]/5 + np.pi*t) 
		elif x[0] == n-1:
			return 0.
		return None
	eq.set_constraints(dirichlet=dirichlet)
	return eq

def edge_diffusion(m, n, constr, periodic=False):
	# G = lattice45(10, 11)
	G, (l, r, t, b) = constr(m, n, with_boundaries=True)
	if periodic:
		for x in l.nodes:
			for y in r.nodes:
				if x[1] == y[1]:
					G.add_edge(x, y)
	# G = nx.hexagonal_lattice_graph(10, 11)
	# G = grid_graph(10, 11, diagonals=True)
	v = gds.edge_gds(G)
	v.set_evolution(dydt=lambda t, y: v.laplacian(y))
	def boundary(e):
		if e[0] in l.nodes and e[1] in r.nodes and e[0][1] == 0:
			return -1.0
		if e in b.edges:
			return 1.0
		elif e in t.edges:
			return 0.0
		return None
	v.set_constraints(dirichlet=boundary)
	return v

if __name__ == '__main__':
	# eq = sinus_boundary_timevarying()
	eq1 = edge_diffusion(10, 22, gds.triangular_lattice, periodic=True)
	eq2 = edge_diffusion(10, 12, gds.square_lattice, periodic=True)

	sys = gds.couple({
		'Edge diffusion on a triangular lattice': eq1,
		'Edge diffusion on a square lattice': eq2,
	})

	gds.render(sys)
