'''
Lattice symmetries in Navier-Stokes
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
from .fluid import incompressible_ns_flow

''' Systems ''' 

def sq_couette_ivp(viscosity, density):
	m, n = 11, 10
	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True, with_lattice_components=True)
	faces, outer_face = gds.embedded_faces(G)
	for j in range(m):
		G.add_edge((n-1, j), (0, j))
	# for i in range(n):
	# 	G.add_edge((i, 0), (i, m-1))
	aux_faces = [((n-1, j), (0, j), (0, j+1), (n-1, j+1)) for j in range(m-1)]
	# aux_faces += [((i, 0), (i, m-1), (i+1, m-1), (i+1, 0)) for i in range(n-1)]
	G.faces = faces + aux_faces # Hacky
	G.rendered_faces = np.array(range(len(faces)), dtype=np.intp) # Hacky

	pressure, velocity = incompressible_ns_flow(G, viscosity=viscosity, density=density)
	vel = 1.0
	def walls(e):
		if e[0][1] == e[1][1] == m//2:
			if e[0][0] == 0 and e[1][0] == n-1:
				return -vel
			return vel
		return 0
	velocity.set_initial(y0=walls)
	return pressure, velocity

def tri_couette_ivp(viscosity, density):
	m, n = 10, 20
	G, (l, r, t, b) = gds.triangular_lattice(m, n, with_boundaries=True, with_lattice_components=True)
	lcomps = G.lattice_components
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

	# hacky
	nodes, edges = set(G.nodes()), set(G.edges())
	for comp in lcomps.values():
		comp.remove_nodes_from(set(comp.nodes()) - nodes)
		comp.remove_edges_from(set(comp.edges() - edges))
	G.lattice_components = lcomps 

	pressure, velocity = incompressible_ns_flow(G, viscosity=viscosity, density=density)
	vel = 1.0
	def walls(e):
		if e[0][1] == e[1][1] == m//2:
			if e == ((0, 0), (n//2-1, 0)):
				return -vel
			return vel
		return 0
	velocity.set_initial(y0=walls)
	return pressure, velocity

def hex_couette_ivp(viscosity, density):
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

	pressure, velocity = incompressible_ns_flow(G, viscosity=viscosity, density=density)
	vel = 1.0
	def walls(e):
		if (e[0][1] == e[1][1] == n//2) and (e[0][0] == e[1][0] - 1):
			return vel
		return 0
	velocity.set_initial(y0=walls)
	return pressure, velocity


''' Testing functions ''' 

def render1():
	viscosity, density = 1., 1e-2
	p, v = sq_couette_ivp(viscosity, density)
	# p, v = tri_couette_ivp(viscosity, density)
	# p, v = hex_couette_ivp(viscosity, density)


	# components = dict()
	# for (name, G) in p.G.lattice_components.items():
	# 	components[name] = v.project(G, lambda v: v)
	# 	components[f'{name}_energy'] = components[name].project(PointObservable, lambda v: (v.y ** 2).sum(), min_rng=0.1)
	# 	components[f'{name}_momentum'] = components[name].project(PointObservable, lambda v: v.y.sum(), min_rng=0.1)

	sys = gds.couple({
		'velocity': v,
		'pressure': p,
		'vorticity': v.project(GraphDomain.faces, lambda v: v.curl()),
		'mass flux': v.project(GraphDomain.edges, lambda v: viscosity * v.laplacian() - p.grad()), 
		'total mass flux': v.project(PointObservable, lambda v: np.abs(viscosity * v.laplacian() - p.grad()).sum()),
		'divergence': v.project(GraphDomain.nodes, lambda v: v.div()),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True, min_rng_size=1e-2)

def render2():
	viscosity, density = 1., 1e-2
	p1, v1 = sq_couette_ivp(viscosity, density)
	p2, v2 = tri_couette_ivp(viscosity, density)
	p3, v3 = hex_couette_ivp(viscosity, density)


	sys = gds.couple({
		'velocity_sq': v1,
		'velocity_tri': v2,
		'velocity_hex': v3,
		'flow_material_derivative_sq': v1.project(PointObservable, lambda v: np.abs(viscosity * v.laplacian() - p1.grad()).sum()),
		'flow_material_derivative_tri': v2.project(PointObservable, lambda v: np.abs(viscosity * v.laplacian() - p2.grad()).sum()),
		'flow_material_derivative_hex': v3.project(PointObservable, lambda v: np.abs(viscosity * v.laplacian() - p3.grad()).sum()),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True, min_rng_size=1e-2)


if __name__ == '__main__':
	render2()