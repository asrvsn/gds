''' Laplace equation, or steady-state heat heat transfer ''' 

from gpde import *
from gpde.render.bokeh import *

def sinus_boundary() -> vertex_pde:
	n = 10
	G = nx.grid_2d_graph(n, n)
	eq = vertex_pde(G, lhs=lambda t, self: self.laplacian())
	def dirichlet(t, x):
		if x[0] == 0:
			return np.sin(x[1]/10)
		return None
	eq.set_boundary(dirichlet=dirichlet, dynamic=False)
	return eq

def sinus_boundary_timevarying() -> vertex_pde:
	n = 10
	G = nx.grid_2d_graph(n, n)
	eq = vertex_pde(G, lhs=lambda t, self: self.laplacian())
	def dirichlet(t, x):
		if x[0] == 0:
			return np.sin(x[1]/5 + t) ** 2
		return None
	eq.set_boundary(dirichlet=dirichlet)
	return eq

if __name__ == '__main__':
	eq = sinus_boundary_timevarying()
	SingleRenderer(eq.system()).start()