''' Poisson equation ''' 

from gpde import *
from gpde.render.bokeh import *

def cubic_f() -> vertex_pde:
	n = 15
	c = 1.0
	G = nx.grid_2d_graph(n, n)
	eq = vertex_pde(G, lhs=lambda t, self: self.laplacian() + c*self.y**3)
	eq.set_boundary(dirichlet=lambda t, x: 0., dynamic=False)
	eq.set_initial(y0=lambda _: 1.)
	return eq

if __name__ == '__main__':
	eq = cubic_f()
	SingleRenderer(eq.system()).start()