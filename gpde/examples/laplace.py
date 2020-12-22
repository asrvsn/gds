''' Laplace equation, or steady-state heat heat transfer ''' 

import pdb
from gpde import *
from gpde.render.bokeh import *

def sinus_boundary() -> vertex_pde:
	n = 10
	G = nx.grid_2d_graph(n, n)
	eq = vertex_pde(G, lhs=lambda t, self: self.laplacian())
	def dirichlet(x):
		if x[0] == 0:
			return np.sin(x[1]/10)
		return None
	eq.set_boundary(dirichlet=dirichlet)
	return eq

def sinus_boundary_timevarying() -> vertex_pde:
	n = 10
	G = nx.grid_2d_graph(n, n)
	eq = vertex_pde(G, lhs=lambda t, self: self.laplacian(), gtol=1e-8)
	def dirichlet(t, x):
		if x[0] == 0:
			return np.sin(x[1]/5 + np.pi*t) 
		elif x[0] == n-1:
			return 0.
		return None
	eq.set_boundary(dirichlet=dirichlet, dynamic=True)
	return eq

if __name__ == '__main__':
	eq = sinus_boundary_timevarying()
	renderer = LiveRenderer(eq.system('laplace'), single_canvas(eq), node_palette=cc.rainbow, node_rng=(-1,1), edge_max=0.3, node_size=0.03)
	renderer.start()
