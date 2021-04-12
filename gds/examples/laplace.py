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
	G, (l, r, t, b) = constr(m, n, with_boundaries=True)
	if periodic:
		for x in l.nodes:
			for y in r.nodes:
				if x[1] == y[1]:
					G.add_edge(x, y)
	v = gds.edge_gds(G)
	v.set_evolution(dydt=lambda t, y: v.laplacian(y))
	velocity = -1.0
	def boundary(e):
		if e[0] in l.nodes and e[1] in r.nodes and e[0][1] == 0:
			return -velocity
		if e in b.edges:
			if e[1][1] > e[0][1] and e[0][0] % 2 == 0:
				# Hack to fix hexagonal bcs
				return -velocity
			return velocity
		elif e in t.edges:
			return 0.0
		return None
	v.set_constraints(dirichlet=boundary)
	return v

def edge_diffusion_comp():
	eq1 = edge_diffusion(10, 22, gds.triangular_lattice, periodic=False)
	curl1 = eq1.project(gds.GraphDomain.faces, lambda eq: eq.curl())
	eq2 = edge_diffusion(10, 12, gds.square_lattice, periodic=False)
	curl2 = eq2.project(gds.GraphDomain.faces, lambda eq: eq.curl())
	eq3 = edge_diffusion(10, 12, gds.hexagonal_lattice, periodic=False)
	curl3 = eq3.project(gds.GraphDomain.faces, lambda eq: eq.curl())
	sys = gds.couple({
		'flow1': eq1,
		'curl1': curl1,
		'flow2': eq2,
		'curl2': curl2,
		'flow3': eq3,
		'curl3': curl3,
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 2), edge_max=0.6)


def square_lattice_edge_diffusion():
	G, (l, r, t, b) = gds.square_lattice(10, 10, with_boundaries=True)
	v = gds.edge_gds(G)
	v.set_evolution(dydt=lambda t, y: v.laplacian(y))
	velocity = 1.0
	def boundary(e):
		if e in b.edges:
			return velocity
		elif e in t.edges:
			return 0.0
		return None
	v.set_constraints(dirichlet=boundary)
	sys = gds.couple({
		'velocity': v,
		'divergence': v.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'curl': v.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'curl*curl': v.project(gds.GraphDomain.edges, lambda v: v.curl_face.T@v.curl_face@v.y),
	})
	gds.render(sys, edge_max=0.6, dynamic_ranges=True)

def test_curl():
	G1 = nx.Graph()
	G1.add_nodes_from([1, 2, 3, 4])
	G1.add_edges_from([(1,2), (2,3), (3, 4), (4, 1)])
	v1 = gds.edge_gds(G1)
	v1.set_evolution(nil=True)
	v1.set_initial(y0=lambda e: 1 if e == (1,2) else 0)
	G2 = nx.Graph()
	G2.add_nodes_from([1, 2, 3, 4, 5, 6])
	G2.add_edges_from([(1,2), (2,3), (3, 4), (4, 1), (1, 5), (5, 6), (6, 2)])
	v2 = gds.edge_gds(G2)
	v2.set_evolution(nil=True)
	v2.set_initial(y0=lambda e: 1 if e == (1,2) else 0)
	sys = gds.couple({
		'velocity1': v1,
		'curl1': v1.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'curl*curl1': v1.project(gds.GraphDomain.edges, lambda v: v.curl_face.T@v.curl_face@v.y),
		'velocity2': v2,
		'curl2': v2.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'curl*curl2': v2.project(gds.GraphDomain.edges, lambda v: v.curl_face.T@v.curl_face@v.y)
	})
	gds.render(sys, edge_max=0.5, canvas=gds.grid_canvas(sys.observables.values(), 3), dynamic_ranges=True)

if __name__ == '__main__':
	square_lattice_edge_diffusion()
	# test_curl()
