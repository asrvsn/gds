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


def square_edge_diffusion():
	m, n = 10, 10
	G, (l, r, t, b) = gds.square_lattice(n, m, with_boundaries=True)
	faces, outer_face = gds.embedded_faces(G)
	for j in range(m):
		G.add_edge((n-1, j), (0, j))
	aux_faces = [((n-1, j), (0, j), (0, j+1), (n-1, j+1)) for j in range(m-1)]
	G.faces = faces + aux_faces # Hacky
	G.rendered_faces = np.array(range(len(faces)), dtype=np.intp) # Hacky
	v = gds.edge_gds(G)
	v.set_evolution(dydt=lambda t, y: v.laplacian(y))
	velocity = 1.0
	def boundary(e):
		if e in b.edges:
			return velocity
		elif e == ((0, 0), (m-1, 0)):
			return -velocity
		elif e in t.edges or e == ((0, n-1), (m-1, n-1)):
			return 0.0
		return None
	v.set_constraints(dirichlet=boundary)
	return v

def tri_edge_diffusion():
	m, n = 10, 20
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True)
	faces, outer_face = gds.embedded_faces(G)
	for j in range(m+1):
		G = nx.algorithms.minors.contracted_nodes(G, (0, j), ((n + 1) // 2, j))
	rendered_faces = set()
	r_nodes = set(r.nodes())
	for i, face in enumerate(faces):
		face = list(face)
		modified = False
		for j, node in enumerate(face):
			if node in r_nodes:
				n_l = (0, node[1]) # identified
				face[j] = n_l
				faces[i] = tuple(face)
				modified = True
		if not modified:
			rendered_faces.add(i)
	G.faces = faces
	G.rendered_faces = np.array(sorted(list(rendered_faces)), dtype=np.intp) # Hacky
	v = gds.edge_gds(G)
	v.set_evolution(dydt=lambda t, y: v.laplacian(y))
	velocity = 1.0
	def boundary(e):
		if e in b.edges:
			return velocity
		elif e == ((0, 0), (n//2-1, 0)):
			return -velocity
		elif e in t.edges or e == ((0, m), (n//2-1, m)):
			return 0.0
		return None
	v.set_constraints(dirichlet=boundary)
	return v

def hex_edge_diffusion():
	m, n = 10, 20
	G, (l, r, t, b) = gds.hexagonal_lattice(m, n, with_boundaries=True)
	faces, outer_face = gds.embedded_faces(G)
	contractions = {}
	for j in range(1, 2*m+1):
		G = nx.algorithms.minors.contracted_nodes(G, (0, j), (n, j))
		contractions[(n, j)] = (0, j)
	nx.set_node_attributes(G, None, 'contraction')
	rendered_faces = set()
	for i, face in enumerate(faces):
		face = list(face)
		modified = False
		for j, node in enumerate(face):
			if node in contractions:
				n_l = contractions[node] # identified
				face[j] = n_l
				faces[i] = tuple(face)
				modified = True
		if not modified:
			rendered_faces.add(i)
	G.faces = faces
	G.rendered_faces = np.array(sorted(list(rendered_faces)), dtype=np.intp) # Hacky
	# pdb.set_trace()
	v = gds.edge_gds(G)
	v.set_evolution(dydt=lambda t, y: v.laplacian(y))
	velocity = 1.0
	def boundary(e):
		if (e[0][1] == e[1][1] == 0) and (e[0][0] == e[1][0] - 1):
			return velocity
		elif e in t.edges or e == ((0, 2*m), (n, 2*m+1)):
			return 0.0
		return None
	v.set_constraints(dirichlet=boundary)
	return v

def render_edge_diffusion(v):
	sys = gds.couple({
		'velocity': v,
		'divergence': v.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'curl': v.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'd*d': v.project(gds.GraphDomain.edges, lambda v: -v.curl_face.T@v.curl_face@v.y),
		'dd*': v.project(gds.GraphDomain.edges, lambda v: v.dirichlet_laplacian@v.y),
		'L1': v.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
	})
	gds.render(sys, edge_max=0.6, dynamic_ranges=True, canvas=gds.grid_canvas(sys.observables.values(), 3))


def edge_diffusion_comp():
	v1 = square_edge_diffusion()
	v2 = tri_edge_diffusion()
	v3 = hex_edge_diffusion()
	sys = gds.couple({
		'flow_square': v1,
		'flow_tri': v2,
		'flow_hex': v3,
		'curl_square': v1.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'curl_tri': v2.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'curl_hex': v3.project(gds.GraphDomain.faces, lambda v: v.curl()),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True)

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
	# render_edge_diffusion(hex_edge_diffusion())
	# test_curl()
	edge_diffusion_comp()
