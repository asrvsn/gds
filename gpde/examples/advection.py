import networkx as nx

from gpde.core import *
from gpde.render.bokeh import *

def advection_on_torus():
	n = 40
	G = nx.grid_2d_graph(n, n, periodic=True)
	def v_field(e: Edge):
		if e[0][1] == e[1][1]:
			if e[0][0] > e[1][0] or e[1][0] == (e[0][0] - n - 1):
				return 1
			else:
				return -1
		else:
			return 0
	concentration = vertex_pde(G, lambda t, self: self.advect(v_field))
	concentration.set_initial(y0 = lambda t, x: 1.0 if x == (10, 10) else None) # delta initial condition
	return concentration

if __name__ == '__main__':
	sys = advection_on_torus().system()
	render_bokeh(GridRenderer(sys))
