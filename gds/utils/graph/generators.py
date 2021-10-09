''' Graph utilities ''' 
import networkx as nx
import numpy as np
import pdb
import matplotlib.pyplot as plt
from typing import Union, Callable
from shapely.geometry import Point, Polygon
import warnings
import trimesh as tm
import itertools

from .voronoi import voronoi

''' Graph generators ''' 

def set_grid_graph_layout(G: nx.Graph):
	m, n = 0, 0
	nodes = set(G.nodes())
	for node in nodes:
		n = max(node[0], n)
		m = max(node[1], m)
	m += 1
	n += 1
	layout = dict()
	dh = 1/max(m, n)
	x0 = -n/max(m, n)
	y0 = -m/max(m, n)
	for i in range(n):
		for j in range(m):
			if (i, j) in nodes:
				layout[(i, j)] = np.array([2*i*dh + x0, 2*j*dh + y0])
	nx.set_node_attributes(G, layout, 'pos')

def square_lattice(m: int, n: int, diagonals=False, with_boundaries=False, with_lattice_components=False, **kwargs) -> nx.Graph:
	G = nx.grid_2d_graph(n, m, **kwargs)
	set_grid_graph_layout(G)
	if diagonals:
		for i in range(n-1):
			for j in range(m-1):
				G.add_edges_from([((i, j), (i+1, j+1)), ((i, j+1), (i+1, j))])
	if with_lattice_components:
		vert = nx.Graph()
		for j in range(n):
			sub = G.subgraph([(j, i) for i in range(m)])
			vert = nx.compose(vert, sub.copy())
		horiz = nx.Graph()
		for i in range(m):
			sub = G.subgraph([(j, i) for j in range(n)])
			horiz = nx.compose(horiz, sub.copy())
		G.lattice_components = {'vert': vert, 'horiz': horiz}
	if with_boundaries:
		l = G.subgraph([(0, i) for i in range(m)])
		r = G.subgraph([(n-1, i) for i in range(m)])
		t = G.subgraph([(j, m-1) for j in range(n)])
		b = G.subgraph([(j, 0) for j in range(n)])
		return G, (l.copy(), r.copy(), t.copy(), b.copy())
	else:
		return G

def square_cylinder(m, n) -> nx.Graph:
	'''
	Y-periodic triangular lattice
	'''
	G = nx.grid_2d_graph(n, m)
	set_grid_graph_layout(G)
	G.l_boundary = G.subgraph([(0, i) for i in range(m)])
	G.r_boundary = G.subgraph([(n-1, i) for i in range(m)])
	# G.faces, _ = embedded_faces(G)
	# for j in range(n):
	# 	G.add_edge((j, m-1), (j, 0))
		# if j < n-1:
		# 	G.faces.append((((j, m-1), (j, 0), (j+1, 0), (j+1, m-1))))
	return G

def diagonal_lattice(m: int, n: int) -> nx.Graph:
	''' Creates 45-degree rotated square lattice; make n odd for symmetric boundaries '''
	G = nx.Graph()
	layout = dict()
	dy = 2/(m-1)
	dx = 2/(n-1)
	for i in range(n):
		if i % 2 == 0:
			for j in range(m-1):
				G.add_node((i, j))
				layout[(i, j)] = np.array([-1+i*dx, -1+dy/2+j*dy])
				if i > 0:
					G.add_edges_from([((i-1, j), (i, j)), ((i-1, j+1), (i, j))])
		else:
			for j in range(m):
				G.add_node((i, j))
				layout[(i, j)] = np.array([-1+i*dx, -1+j*dy])
				if j > 0:
					G.add_edges_from([((i-1, j-1), (i, j-1)), ((i-1, j-1), (i, j))])
	nx.set_node_attributes(G, layout, 'pos')
	return G

def triangular_lattice(m, n, with_boundaries=False, with_lattice_components=False, **kwargs) -> nx.Graph:
	''' Sanitize networkx properties for Bokeh consumption ''' 
	if 'periodic' in kwargs:
		kwargs['with_positions'] = False
		G = nx.triangular_lattice_graph(m, n, **kwargs)
		nx.set_node_attributes(G, None, 'contraction')
		return G
	else:
		G = nx.triangular_lattice_graph(m, n, **kwargs)
		if with_lattice_components:
			horiz = nx.Graph()
			for i in range(m+1):
				sub = G.subgraph([(j, i) for j in range(n)])
				horiz = nx.compose(horiz, sub.copy())
			diag_l = G.copy()
			for i in range(m+1):
				remove_edges(diag_l, [((j, i), (j+1,i)) for j in range(n)])
			diag_r = diag_l.copy()
			for i in range(m+1):
				if i % 2 == 0:
					remove_edges(diag_l, [((j, i), (j, i+1)) for j in range(n)])
					remove_edges(diag_r, [((j, i), (j-1, i+1)) for j in range(n)])
				else:
					remove_edges(diag_l, [((j, i), (j+1, i+1)) for j in range(n)])
					remove_edges(diag_r, [((j, i), (j, i+1)) for j in range(n)])
			G.lattice_components = {'diag_l': diag_l, 'diag_r': diag_r, 'horiz': horiz}
		if with_boundaries:
			l = G.subgraph([(0, i) for i in range(m+1)])
			r_nodes = [(n//2, 2*i+1) for i in range(m//2+1)]
			if n % 2 == 1:
				r_nodes += [(n//2+1, i) for i in range(m+1)]
			else:
				r_nodes += [(n//2, 2*i) for i in range(m//2+1)]
			r = G.subgraph([x for x in r_nodes if x in G.nodes])
			t = G.subgraph([(j, m) for j in range(n)])
			b = G.subgraph([(j, 0) for j in range(n)])
			return G, (l.copy(), r.copy(), t.copy(), b.copy())
		else:
			return G

def triangular_cylinder(m, n) -> nx.Graph:
	'''
	Y-periodic triangular lattice
	'''
	G = nx.triangular_lattice_graph(m, n)
	N = (n + 1) // 2  # number of nodes in row
	cols = range(N + 1)
	for i in cols:
		G = nx.contracted_nodes(G, (i, 0), (i, m))
	nx.set_node_attributes(G, None, 'contraction')
	G.l_boundary = G.subgraph([(0, i) for i in range(m)])
	r_nodes = [(n//2, 2*i+1) for i in range(m//2+1)]
	if n % 2 == 1:
		r_nodes += [(n//2+1, i) for i in range(m+1)]
	else:
		r_nodes += [(n//2, 2*i) for i in range(m//2+1)]
	G.r_boundary = G.subgraph([x for x in r_nodes if x in G.nodes])
	G.faces = find_k_cliques(G, 3)
	return G

def hexagonal_lattice(m, n, with_boundaries=False, with_lattice_components=False, with_faces=True, **kwargs) -> nx.Graph:
	''' Sanitize networkx properties for Bokeh consumption ''' 
	if 'periodic' in kwargs:
		kwargs['with_positions'] = False
		G = nx.hexagonal_lattice_graph(m, n, **kwargs)
		nx.set_node_attributes(G, None, 'contraction')
		return G
	else:
		G = nx.hexagonal_lattice_graph(m, n, **kwargs)
		if with_faces:
			G.faces, _ = embedded_faces(G)
		if with_lattice_components:
			horiz = nx.Graph()
			for i in range(m+1):
				sub = G.subgraph([(j, i) for j in range(n)])
				horiz = nx.compose(horiz, sub.copy())
			diag_l = G.copy()
			for i in range(m+1):
				remove_edges(diag_l, [((j, i), (j+1,i)) for j in range(n)])
			diag_r = diag_l.copy()
			for i in range(m+1):
				if i % 2 == 0:
					remove_edges(diag_l, [((j, i), (j+1, i)) for j in range(n)])
					remove_edges(diag_r, [((j, i), (j+1, i-1)) for j in range(n)])
				else:
					remove_edges(diag_l, [((j, i), (j+1, i+1)) for j in range(n)])
					remove_edges(diag_r, [((j, i), (j, i+1)) for j in range(n)])
			G.lattice_components = {'diag_l': diag_l, 'diag_r': diag_r, 'horiz': horiz}
		if with_boundaries:
			l = G.subgraph([(0, i) for i in range(m*2+2)])
			r = G.subgraph([(n, i) for i in range(m*2+2)])
			t = G.subgraph([(j, m*2) for j in range(n+1)] + [(j, m*2+1) for j in range(n+1)])
			b = G.subgraph([(j, 0) for j in range(n+1)] + [(j, 1) for j in range(n+1)])
			return G, (l.copy(), r.copy(), t.copy(), b.copy())
		else:
			return G

def random_planar_graph(n, dist):
	assert n > 0
	G = nx.random_geometric_graph(n, dist)
	G = planarize(G)
	largest_cc = max(nx.connected_components(G), key=len)
	G = G.subgraph(largest_cc).copy()
	return G

def voronoi_lattice(n_boundary: int, n_interior: int, eps=0.05, with_boundaries=False):
	assert 0 < eps < 1
	box = np.array([0., 1., 0., 1.])
	points = np.vstack((
		np.vstack((np.zeros(n_boundary)+eps/2, np.linspace(0+eps/2, 1-eps/2, n_boundary))).T,
		np.vstack((np.ones(n_boundary)-eps/2, np.linspace(0+eps/2, 1-eps/2, n_boundary))).T,
		np.random.uniform(0+eps, 1-eps, (n_interior, 2)),
	))
	vor = voronoi(points, box)
	G = nx.Graph()
	seen = set()
	for region in vor.filtered_regions:
		for node in region:
			if not (node in seen):
				G.add_node(node)
				G.nodes[node]['pos'] = tuple(vor.vertices[node].tolist())
		G.add_edges_from(zip(region[:-1], region[1:]))
		G.add_edge(region[-1], region[0])
	# pdb.set_trace()
	if with_boundaries:
		l = G.subgraph([n for n in G.nodes() if 0-eps/4 <= G.nodes[n]['pos'][0] <= eps/4])
		r = G.subgraph([n for n in G.nodes() if 1-eps/4 <= G.nodes[n]['pos'][0] <= 1+eps/4])
		b = G.subgraph([n for n in G.nodes() if 0-eps/4 <= G.nodes[n]['pos'][1] <= eps/4])
		t = G.subgraph([n for n in G.nodes() if 1-eps/4 <= G.nodes[n]['pos'][1] <= 1+eps/4])
		return G, (l.copy(), r.copy(), t.copy(), b.copy())
	else:
		return G

def flat_prism(k=2, n=4):
	G = nx.Graph()
	layout = dict()
	for i in range(k):
		cycle = [i*n + x for x in range(n)]
		G.add_edges_from(zip(cycle, cycle[-1:] + cycle[:-1]))
		if i > 0:
			cycle_ = [(i-1)*n + x for x in range(n)]
			G.add_edges_from(zip(cycle, cycle_))
		for j, v in enumerate(cycle):
			layout[v] = (i+1) * np.array([np.cos(2*np.pi*(j-1)/n), np.sin(2*np.pi*(j-1)/n)])
	nx.set_node_attributes(G, layout, 'pos')
	return G

def octagonal_prism():
	edges = {
		(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (1,8),
		(9,10), (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), (9,16),
		(1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15), (8,16), 
	}
	G.add_edges_from(edges)
	layout = dict()
	for v in range(1,9):
		layout[v] = np.array([np.cos(2*np.pi*(v-1)/8), np.sin(2*np.pi*(v-1)/8)])
	for v in range(9,17):
		layout[v] = np.array([2*np.cos(2*np.pi*(v-1)/8), 2*np.sin(2*np.pi*(v-1)/8)])
	nx.set_node_attributes(G, layout, 'pos')
	return G

def uv_sphere():
	n = 20
	r = 1.0
	twopi = 2*np.pi
	delta = twopi / n
	nodes = []
	edges = []
	rnd = lambda x: x

	for i in range(n):
		for j in range(n):
			theta = delta * i
			phi = delta * j

			x = r*np.cos(theta)*np.cos(phi)
			y = r*np.sin(theta)*np.cos(phi)
			z = r*np.sin(phi)
			nodes.append(((rnd(x),rnd(y),rnd(z))))

			for i_ in [(i + 1) % n, (i - 1) % n]:
				theta_ = delta * i_
				x_ = r*np.cos(theta_)*np.cos(phi)
				y_ = r*np.sin(theta_)*np.cos(phi)
				z_ = r*np.sin(phi)
				edges.append(((rnd(x),rnd(y),rnd(z)), (rnd(x_),rnd(y_),rnd(z_))))

			for j_ in [(j + 1) % n, (j - 1) % n]:
				phi_ = delta * j_
				x_ = r*np.cos(theta)*np.cos(phi_)
				y_ = r*np.sin(theta)*np.cos(phi_)
				z_ = r*np.sin(phi_)
				edges.append(((rnd(x),rnd(y),rnd(z)), (rnd(x_),rnd(y_),rnd(z_))))

	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	return G

def torus(m=10, n=15):
	'''
	m (poloidal)
	n (toroidal)
	'''
	G = nx.grid_2d_graph(n, m)
	faces, outer_face = embedded_faces(G)
	# Add periodic faces
	faces.extend([((n-1, j), (0, j), (0, j+1), (n-1, j+1)) for j in range(m-1)])
	faces.extend([((i, m-1), (i, 0), (i+1, 0), (i+1, m-1)) for i in range(n-1)])
	G = nx.grid_2d_graph(n, m, periodic=True)
	G.faces = faces
	return G

def k_torus(k=2, m=10, n=15, degenerate=False):
	'''
	k-hole torus
	'''
	assert k > 0
	N = m*n
	all_faces = dict()

	def relabel_node(key, i):
		return N*i + key[0]*m + key[1]
	def relabel_graph(G, i):
		nx.relabel_nodes(G, {key: relabel_node(key, i) for key in G.nodes()}, copy=False)
		return G
	def add_faces(fs, i):
		for f in fs:
			f_ = tuple(relabel_node(v, i) for v in f)
			all_faces[frozenset(f_)] = f_
	def get_face(f, i):
		f_ = tuple(relabel_node(v, i) for v in f)
		return all_faces[frozenset(f_)]
	def remove_faces(fs, i):
		for f in fs:
			f_ = tuple(relabel_node(v, i) for v in f)
			del all_faces[frozenset(f_)]
	def remap_face(f, mp):
		f = list(f)
		for i, v in enumerate(f):
			if v in mp:
				f[i] = mp[v]
		return tuple(f)

	G = relabel_graph(torus(m=m, n=n), 0)
	add_faces(G.faces, 0)

	l_face = ((0,0), (1,0), (1,1), (0,1))
	rn, rm = n//2, m//2
	r_face = ((rn,rm), (rn,rm+1), (rn+1,rm+1), (rn+1,rm))

	contracted = dict()
	for i in range(1, k):
		G_ = relabel_graph(torus(m=m, n=n), i)
		add_faces(G_.faces, i)
		# pdb.set_trace()
		G = nx.union(G, G_)
		if degenerate:
			u_, v_ =  relabel_node(l_face[0], i-1), relabel_node(r_face[0], i)
			G.add_edge(u_, v_)
			# nx.contracted_nodes(G, u_, v_, copy=False)
			# contracted[v_] = u_
		else:
			for (u, v) in zip(l_face, r_face):
				u_ = relabel_node(u, i-1)
				v_ = relabel_node(v, i)
				# G.add_edge(u_, v_)
				nx.contracted_nodes(G, u_, v_, copy=False)
				contracted[v_] = u_
			remove_faces([l_face], i-1)
			remove_faces([r_face], i)

	G.faces = list(map(lambda f: remap_face(f, contracted), all_faces.values()))
	return G

''' Triangulated manifolds ''' 

def icosphere():
	mesh = tm.creation.icosphere(subdivisions=2)
	G = tm.graph.vertex_adjacency_graph(mesh)
	G.faces = find_k_cliques(G, 3)
	return G

def icotorus(m=10, n=15):
	G = triangular_lattice(m, n, periodic=True)
	G.faces = find_k_cliques(G, 3)
	return G

''' Graph helpers ''' 

def get_planar_boundary(G: nx.Graph) -> (nx.Graph, nx.Graph, nx.Graph, nx.Graph, nx.Graph):
	''' Get boundary of planar graph using layout coordinates. ''' 
	nodes = set(G.nodes())
	edges = set(G.edges())
	pos = nx.get_node_attributes(G, 'pos')
	xrng, yrng = list(set([pos[n][0] for n in nodes])), list(set([pos[n][1] for n in nodes]))
	xmin = dict(zip(yrng, [min([pos[n][0] for n in nodes if pos[n][1]==y]) for y in yrng]))
	ymin = dict(zip(xrng, [min([pos[n][1] for n in nodes if pos[n][0]==x]) for x in xrng]))
	xmax = dict(zip(yrng, [max([pos[n][0] for n in nodes if pos[n][1]==y]) for y in yrng]))
	ymax = dict(zip(xrng, [max([pos[n][1] for n in nodes if pos[n][0]==x]) for x in xrng]))
	dG, dG_L, dG_R, dG_T, dG_B = nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()
	for n in nodes:
		x, y = pos[n]
		if x == xmin[y]:
			dG_L.add_node(n)
			dG.add_node(n)
		if x == xmax[y]:
			dG_R.add_node(n)
			dG.add_node(n)
		if y == ymin[x]:
			dG_B.add_node(n)
			dG.add_node(n)
		if y == ymax[x]:
			dG_T.add_node(n)
			dG.add_node(n)
	for _dG in (dG, dG_L, dG_R, dG_T, dG_B):
		for n in _dG.nodes():
			for m in _dG.nodes():
				# Preserve implicit orientation
				if (n, m) in edges:
					_dG.add_edge(n, m)
				elif (m, n) in edges:
					_dG.add_edge(m, n)
	return (dG, dG_L, dG_R, dG_T, dG_B)

def clear_attributes(G):
	ns = list(G.nodes(data=True))
	es = list(G.edges(data=True))
	if len(ns) > 0:
		n = ns[0]
		for attr in n[-1].keys():
			nx.set_node_attributes(G, None, attr)
	if len(es) > 0:
		e = es[0]
		for attr in e[-1].keys():
			nx.set_node_attributes(G, None, attr)
	return G

def embedded_faces(G, use_spring_default=True):
	'''
	Returns the faces of a graph G by embedding within a 2-manifold of minimal genus. 
	- Currently works only for planar graphs (LOL) 
	- Orients planar faces in CCW direction
	- extension to non-zero genus requires a graph embedding (TODO)
	'''
	pos =  nx.get_node_attributes(G, 'pos')
	if pos != {}:
		# Position data already calculated (e.g. in planar graph generators)
		def normalize_angle(theta):
			sign, magn = np.sign(theta), np.abs(theta) % (2*np.pi)
			if sign < 0:
				return 2*np.pi - magn
			else:
				return magn

		def incident_angle(e1, e2):
			# Returns the angle measured from e1 to e2, where -pi <= angle <= pi
			x1, y1 = pos[e1[1]][0]-pos[e1[0]][0], pos[e1[1]][1]-pos[e1[0]][1]
			x2, y2 = pos[e2[1]][0]-pos[e2[0]][0], pos[e2[1]][1]-pos[e2[0]][1]
			theta = normalize_angle(np.arctan2(y2, x2) - np.arctan2(y1, x1))
			return (theta-2*np.pi) if theta > np.pi else theta

		faces = []
		half_edges_seen = set()
		for edge in G.edges():
			for half_edge in (edge, (edge[1], edge[0])):
				if not (half_edge in half_edges_seen):
					# "Keep going left" -- uses orientability of plane
					(tail, head) = half_edge
					path = [head]
					while True:
						ccw = None
						ccw_angle = -float('inf')
						for node in G.neighbors(head):
							if node != tail:
								angle = incident_angle((tail, head), (head, node))
								if angle > ccw_angle:
									ccw_angle = angle
									ccw = node
						if ccw == None:
							path = None
							break
						else:
							if ccw == path[0]:
								break
							path.append(ccw)
							tail = head
							head = ccw
					if path != None:
						faces.append(tuple(path))
						half_edges_seen.update([(path[-1], path[0])] + [(path[i-1], path[i]) for i in range(1, len(path))])

		# Determine the outer face
		outer_face = None
		points = [Point(*pos[node]) for node in G.nodes()]
		for i, face in enumerate(faces):
			poly = Polygon([pos[node] for node in face])
			if all([poly.intersects(p) for p in points]):
				outer_face = face
				del faces[i]
				break
		# assert outer_face != None, 'Could not find outer face!'

		# print('\n'.join([repr(f) for f in faces]))
		print(f'Faces: {len(faces)}')
		return faces, outer_face
	else:
		(is_planar, embedding) = nx.algorithms.planarity.check_planarity(G)
		if is_planar:
			if use_spring_default:
				 pos = nx.spring_layout(G, scale=0.9, center=(0,0), iterations=500, seed=1, dim=2)
				 nx.set_node_attributes(G, pos, 'pos')
				 return embedded_faces(G)
			else:
				half_edges_seen = set()
				faces = []
				for edge in G.edges():
					for half_edge in (edge, (edge[1], edge[0])):
						if not (half_edge in half_edges_seen):
							face = embedding.traverse_face(half_edge[0], half_edge[1], mark_half_edges=half_edges_seen)
							faces.append(tuple(face))
				# Set the node positions now that we have used used this embedding
				pos = nx.algorithms.planar_drawing.combinatorial_embedding_to_pos(embedding)
				nx.set_node_attributes(G, pos, 'pos')
				return faces
		else:
			# Try using a force-directed layout in 3 dimensions
			warnings.warn('TODO: no faces for non-planar graphs')
			return [], None

def planarize(G: nx.Graph):
	'''
	Takes a graph with node embeddings in R^2 and removes random edges until planar.
	'''
	pos = nx.get_node_attributes(G, 'pos')
	assert pos != {}, 'planarize() needs a node embedding'

	def on_segment(p, q, r):
		if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
			return True
		return False

	def orientation(p, q, r):
		val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
		if val == 0 : return 0
		return 1 if val > 0 else -1

	def intersects(seg1, seg2):
		p1, q1 = seg1
		p2, q2 = seg2

		o1 = orientation(p1, q1, p2)
		o2 = orientation(p1, q1, q2)
		o3 = orientation(p2, q2, p1)
		o4 = orientation(p2, q2, q1)

		if o1 != o2 and o3 != o4:
			return True

		if o1 == 0 and on_segment(p1, q1, p2) : return True
		if o2 == 0 and on_segment(p1, q1, q2) : return True
		if o3 == 0 and on_segment(p2, q2, p1) : return True
		if o4 == 0 and on_segment(p2, q2, q1) : return True

		return False

	deleted = set()
	for e1 in G.edges():
		for e2 in G.edges():
			if not (
				e1 in deleted or 
				e2 in deleted or 
				e1[0] in e2 or 
				e1[1] in e2
			):
				e1_pos = (pos[e1[0]], pos[e1[1]])
				e2_pos = (pos[e2[0]], pos[e2[1]])
				if intersects(e1_pos, e2_pos):
					deleted.add(e1)

	G.remove_edges_from(deleted)
	return G

def get_edge_lengths(G: nx.Graph):
	'''
	Takes a graph with node embeddings in R^2 and returns the edge lengths.
	'''
	pos = nx.get_node_attributes(G, 'pos')
	assert pos != {}, 'need a node embedding'
	lengths = dict()
	for edge in G.edges():
		lengths[edge] = np.sqrt((pos[edge[0]][0] - pos[edge[1]][0])**2 + (pos[edge[0]][1] - pos[edge[1]][1])**2)
	return lengths

def get_edge_weights(G: nx.Graph, key='weight', default=1.0):
	weights = dict()
	for (u, v, d) in G.edges(data=True):
		if key in d: 
			weights[(u, v)] = d[key]
		else:
			weights[(u, v)] = default
	return weights

def set_edge_weights(G: nx.Graph, w: Union[Callable, float], key='weight'):
	try: 
		value = float(w)
		w = lambda e: value
	except:
		assert callable(w), 'w must be float or callable'
	for u, v in G.edges():
		G[u][v][key] = w((u, v))
	return G

def get_planar_mesh(G: nx.Graph, tol: float=1e-4):
	'''
	Takes a planar graph and returns its edge-weighted approximation of R^2 as mesh coordinates.
	'''
	pos = nx.get_node_attributes(G, 'pos')
	assert pos != {}, 'need a node embedding'
	lengths = get_edge_lengths(G)
	weights = get_edge_weights(G)
	factors = [1 / (weights[k] * lengths[k]) for k in lengths.keys()]
	assert np.abs(min(factors) - max(factors)) <= tol, 'Cannot achieve desired mesh with uniform scaling.'
	# Rescale and translate
	factor = factors[0]
	xmin, ymin = min([pos[k][0] for k in pos.keys()]), min([pos[k][1] for k in pos.keys()])
	real_pos = {k: (factor*(v[0] - xmin), factor*(v[1] - ymin)) for k, v in pos.items()}
	return real_pos

def remove_edges(G: nx.Graph, edges):
	e_set = set(G.edges())
	for e in edges:
		if e in e_set:
			G.remove_edge(*e)
		elif (e[1], e[0]) in e_set:
			G.remove_edge(e[1], e[0])

def find_k_cliques(G, k):
	'''
	https://stackoverflow.com/questions/58775867/what-is-the-best-way-to-count-the-cliques-of-size-k-in-an-undirected-graph-using
	'''
	k_cliques = set()
	for clique in nx.find_cliques(G):
		if len(clique) == k:
			k_cliques.add(tuple(sorted(clique)))
		elif len(clique) > k:
			for mini_clique in itertools.combinations(clique, k):
				k_cliques.add(tuple(sorted(mini_clique)))
	return k_cliques

def remove_face(G, v):
	faces = []
	for f in G.faces:
		if not (v in f):
			faces.append(f)
	G.faces = faces
	G.remove_node(v)

def extract_face(G, v):
	'''
	Construct an ordered chain from neighbors of v.
	'''
	ns = list(G.neighbors(v))
	print(ns)
	face = [ns.pop()]
	k = len(ns)
	while len(ns) > 0:
		print(face)
		found = False
		for i, n in enumerate(ns):
			print(n, G.neighbors(n))
			if face[-1] in G.neighbors(n):
				face.append(n)
				del ns[i]
				found = True
				break
		assert found, 'Neighbors do not form a chain.'
	return tuple(face)

def contract_pairs(G, pairs):
	'''
	Contract nodes with updates to faces & sanitize output.
	'''
	faces = G.faces if hasattr(G, 'faces') else None
	contracted = dict()
	for (u, v) in pairs:
		nx.contracted_nodes(G, u, v, copy=False)
		contracted[v] = u
	nx.set_node_attributes(G, None, 'contraction')
	if faces != None:
		faces_ = []
		for f in faces:
			faces_.append(tuple((contracted[v] if v in contracted else v) for v in f))
		G.faces = faces_
	return G

def remove_pos(G):
	if len(nx.get_node_attributes(G, 'pos')) > 0:
		for n in G.nodes():
			del G.nodes[n]['pos']
	return G

def edge_domain(G, nodes):
	nodes = set(nodes)
	for (u, v) in G.edges():
		if u in nodes and v in nodes: 
			yield (u, v)

if __name__ == '__main__':
	G = hexagonal_lattice(3, 4)
	faces = embedded_faces(G)
	pdb.set_trace()

	nx.draw_spectral(G)

	plt.show()