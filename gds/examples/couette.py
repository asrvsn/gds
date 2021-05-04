'''
Couette (drag-induced) flow
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc

import gds
from gds.types import *
from .fluid import incompressible_ns_flow

''' Systems ''' 

def sq_couette():
	m, n = 11, 10
	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	faces, outer_face = gds.embedded_faces(G)
	for j in range(m):
		G.add_edge((n-1, j), (0, j))
	aux_faces = [((n-1, j), (0, j), (0, j+1), (n-1, j+1)) for j in range(m-1)]
	G.faces = faces + aux_faces # Hacky
	G.rendered_faces = np.array(range(len(faces)), dtype=np.intp) # Hacky

	pressure, velocity = incompressible_ns_flow(G, viscosity=1., density=1e-2)
	vel = 1.0
	def walls(e):
		if e in t.edges or e == ((0, m-1), (n-1, m-1)):
			return 0
		elif e in b.edges: 
			return vel
		elif e == ((0, 0), (n-1, 0)):
			return -vel
		return None
	velocity.set_constraints(dirichlet=walls)
	return pressure, velocity

def tri_couette():
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

	pressure, velocity = incompressible_ns_flow(G, viscosity=1., density=1e-2)
	vel = 1.0
	def walls(e):
		if e in b.edges:
			return vel
		elif e == ((0, 0), (n//2-1, 0)):
			return -vel
		elif e in t.edges or e == ((0, m), (n//2-1, m)):
			return 0.0
		return None
	velocity.set_constraints(dirichlet=walls)
	return pressure, velocity

def hex_couette():
	m, n = 6, 12
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

	pressure, velocity = incompressible_ns_flow(G, viscosity=1., density=1e-2)
	vel = 1.0
	def walls(e):
		if (e[0][1] == e[1][1] == 0) and (e[0][0] == e[1][0] - 1):
			return vel
		elif e in t.edges or e == ((0, 2*m), (n, 2*m+1)):
			return 0.0
		return None
	velocity.set_constraints(dirichlet=walls)
	return pressure, velocity

def sq_couette_ivp():
	m, n = 11, 10
	G, (l, r, t, b) = gds.square_lattice(m, n, with_boundaries=True)
	faces, outer_face = gds.embedded_faces(G)
	for j in range(m):
		G.add_edge((n-1, j), (0, j))
	# for i in range(n):
	# 	G.add_edge((i, 0), (i, m-1))
	aux_faces = [((n-1, j), (0, j), (0, j+1), (n-1, j+1)) for j in range(m-1)]
	# aux_faces += [((i, 0), (i, m-1), (i+1, m-1), (i+1, 0)) for i in range(n-1)]
	G.faces = faces + aux_faces # Hacky
	G.rendered_faces = np.array(range(len(faces)), dtype=np.intp) # Hacky

	pressure, velocity = incompressible_ns_flow(G, viscosity=1., density=1e-2)
	vel = 1.0
	def walls(e):
		if e[0][1] == e[1][1] == m//2:
			if e[0][0] == 0 and e[1][0] == n-1:
				return -vel
			return vel
		return 0
	velocity.set_initial(y0=walls)
	return pressure, velocity

def tri_couette_ivp():
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

	pressure, velocity = incompressible_ns_flow(G, viscosity=1., density=1e-2)
	vel = 1.0
	def walls(e):
		if e[0][1] == e[1][1] == m//2:
			if e == ((0, 0), (n//2-1, 0)):
				return -vel
			return vel
		return 0
	velocity.set_initial(y0=walls)
	return pressure, velocity

''' Testing functions ''' 

def render():
	p1, v1 = sq_couette_ivp()
	# p1, v1 = tri_couette_ivp()
	# p1, v1 = sq_couette()
	# p2, v2 = tri_couette()
	# p3, v3 = hex_couette()

	sys = gds.couple({
		'velocity': v1,
		# 'velocity_tri': v2,
		# 'velocity_hex': v3,
		# 'pressure': p1,
		# 'pressure_tri': p2,
		# 'pressure_hex': p3,
		'vorticity': v1.project(gds.GraphDomain.faces, lambda v: v.curl()),
		# 'vorticity_tri': v2.project(gds.GraphDomain.faces, lambda v: v.curl()),
		# 'vorticity_hex': v3.project(gds.GraphDomain.faces, lambda v: v.curl()),
		# 'divergence_square': v1.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'kinetic energy': v1.project(PointObservable, lambda v: (v1.y ** 2).sum(), min_rng=0.01),
		'momentum': v1.project(PointObservable, lambda v: np.abs(v1.y).sum(), min_rng=0.01),
		'rotational energy': v1.project(PointObservable, lambda v: (v1.curl() ** 2).sum(), min_rng=0.01),
		'angular momentum': v1.project(PointObservable, lambda v: np.abs(v1.curl()).sum(), min_rng=0.01),
	})
	gds.render(sys, canvas=gds.grid_canvas(sys.observables.values(), 3), edge_max=0.6, dynamic_ranges=True, min_rng_size=1e-2)

def dump():
	p1, v1 = sq_couette()
	p2, v2 = tri_couette()
	p3, v3 = hex_couette()

	sys = gds.couple({
		'velocity_square': v1,
		'velocity_tri': v2,
		'velocity_hex': v3,
		'pressure_square': p1,
		'pressure_tri': p2,
		'pressure_hex': p3,
		'vorticity_square': v1.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'vorticity_tri': v2.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'vorticity_hex': v3.project(gds.GraphDomain.faces, lambda v: v.curl()),
		'diffusion_square': v1.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		'diffusion_tri': v2.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		'diffusion_hex': v3.project(gds.GraphDomain.edges, lambda v: v.laplacian()),
		'advection_square': v1.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		'advection_tri': v2.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		'advection_hex': v3.project(gds.GraphDomain.edges, lambda v: -v.advect()),
		'divergence_square': v1.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'divergence_tri': v2.project(gds.GraphDomain.nodes, lambda v: v.div()),
		'divergence_hex': v3.project(gds.GraphDomain.nodes, lambda v: v.div()),
	})

	sys.solve_to_disk(2.0, 0.01, 'couette')

if __name__ == '__main__':
	render()