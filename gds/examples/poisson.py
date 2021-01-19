''' Poisson equation ''' 

import numpy as np

from gds import *
from gds.render.bokeh import *

def poisson_eq(n: int, f: Callable[[np.ndarray], np.ndarray], boundary: Callable[[Time, Point], float]) -> node_gds:
	G = grid_graph(n, n)
	eq = node_gds(G, lhs=lambda t, self: self.laplacian() + f(self.y))
	eq.set_constraints(dirichlet=boundary)
	eq.set_initial(y0=lambda _: 1.)
	return eq

def flat():
	n = 20
	def boundary(t, x):
		if x[0] == 0 or x[1] == 0 or x[0] == n-1 or x[1] == n-1:
			return 0.
		return None
	return poisson_eq(n, lambda y: 1, boundary)

def ell():
	n = 20
	def boundary(t, x):
		if x[0] == 0:
			return (n-x[1]) / n
		elif x[1] == 0:
			return (n-x[0]) / n
		elif x[0] == n-1 or x[1] == n-1:
			return 0.
		return None
	return poisson_eq(n, lambda y: 1.0, boundary)

def ell_quartic():
	n = 20
	def boundary(t, x):
		if x[0] == 0:
			return (n-x[1]) / n
		elif x[1] == 0:
			return (n-x[0]) / n
		elif x[0] == n-1 or x[1] == n-1:
			return 0.
		return None
	return poisson_eq(n, lambda y: y**4, boundary)

if __name__ == '__main__':
	e1 = ell()
	e2 = ell_quartic()
	GridRenderer(couple(e1, e2), n_spring_iters=1000).start()