''' Graph utilities ''' 
import networkx as nx
import numpy as np
import pdb
import matplotlib.pyplot as plt

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

def square_lattice(m: int, n: int, diagonals=False, with_boundaries=False, **kwargs) -> nx.Graph:
	G = nx.grid_2d_graph(n, m, **kwargs)
	set_grid_graph_layout(G)
	if diagonals:
		for i in range(n-1):
			for j in range(m-1):
				G.add_edges_from([((i, j), (i+1, j+1)), ((i, j+1), (i+1, j))])
	if with_boundaries:
		l = G.subgraph([(0, i) for i in range(m)])
		r = G.subgraph([(n-1, i) for i in range(m)])
		t = G.subgraph([(j, m-1) for j in range(n)])
		b = G.subgraph([(j, 0) for j in range(n)])
		return G, (l.copy(), r.copy(), t.copy(), b.copy())
	else:
		return G

def lattice45(m: int, n: int) -> nx.Graph:
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

def triangular_lattice(m, n, with_boundaries=False, **kwargs) -> nx.Graph:
	''' Sanitize networkx properties for Bokeh consumption ''' 
	if 'periodic' in kwargs:
		kwargs['with_positions'] = False
		G = nx.triangular_lattice_graph(m, n, **kwargs)
		nx.set_node_attributes(G, None, 'contraction')
		return G
	else:
		G = nx.triangular_lattice_graph(m, n, **kwargs)
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

def hexagonal_lattice(m, n, with_boundaries=False, **kwargs) -> nx.Graph:
	''' Sanitize networkx properties for Bokeh consumption ''' 
	if 'periodic' in kwargs:
		kwargs['with_positions'] = False
		G = nx.hexagonal_lattice_graph(m, n, **kwargs)
		nx.set_node_attributes(G, None, 'contraction')
		return G
	else:
		G = nx.hexagonal_lattice_graph(m, n, **kwargs)
		if with_boundaries:
			l = G.subgraph([(0, i) for i in range(m*2+2)])
			r = G.subgraph([(n, i) for i in range(m*2+2)])
			t = G.subgraph([(j, m*2) for j in range(n+1)] + [(j, m*2+1) for j in range(n+1)])
			b = G.subgraph([(j, 0) for j in range(n+1)] + [(j, 1) for j in range(n+1)])
			return G, (l.copy(), r.copy(), t.copy(), b.copy())
		else:
			return G


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
		# largest = max([len(f) for f in faces])
		# faces = [f for f in faces if len(f) < largest]
		# print('\n'.join([repr(f) for f in faces]))
		print(f'Faces: {len(faces)}')
		# pdb.set_trace()
		return faces
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
			raise Exception('TODO: implement faces for non-planar graphs')

if __name__ == '__main__':
	G = hexagonal_lattice(3, 4)
	faces = embedded_faces(G)
	pdb.set_trace()

	nx.draw_spectral(G)

	plt.show()