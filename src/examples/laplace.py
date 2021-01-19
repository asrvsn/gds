''' Laplace equation, or steady-state heat heat transfer ''' 

import pdb
from gds import *
from gds.utils.graph import *
from gds.render.bokeh import *

def sinus_boundary() -> node_gds:
	n = 10
	G = grid_graph(n, n)
	eq = node_gds(G, lhs=lambda t, self: self.laplacian())
	def dirichlet(x):
		if x[0] == 0:
			return np.sin(x[1]/10)
		return None
	eq.set_constraints(dirichlet=dirichlet)
	return eq

def sinus_boundary_timevarying() -> node_gds:
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

def edge_diffusion():
	# G = lattice45(10, 11)
	# G = nx.triangular_lattice_graph(10, 11)
	# G = nx.hexagonal_lattice_graph(10, 11)
	G = grid_graph(10, 11, diagonals=True)
	dG, dG_L, dG_R, dG_T, dG_B = get_planar_boundary(G)
	v = edge_gds(G, dydt=lambda t, self: self.laplacian())
	def boundary(e):
		if e in dG_B.edges:
			return 1.0
		elif e in dG_T.edges:
			return 0.0
		return None
	v.set_constraints(dirichlet=boundary)
	return v

if __name__ == '__main__':
	# eq = sinus_boundary_timevarying()
	eq = edge_diffusion()
	dual = eq.dual()

	sys = System(eq, {
		'eq': eq,
		'dual': dual,
	})

	renderer = LiveRenderer(sys, grid_canvas([eq, dual]), node_palette=cc.rainbow, node_rng=(-1, 1), node_size=0.03)
	renderer.start()
