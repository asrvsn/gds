''' Graph utilities ''' 
import networkx as nx
import numpy as np
import pdb

def grid_graph_layout(G: nx.Graph):
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
	return layout

def grid_graph(m: int, n: int, **kwargs) -> nx.Graph:
	G = nx.grid_2d_graph(n, m, **kwargs)
	pos = grid_graph_layout(G)
	nx.set_node_attributes(G, pos, 'pos')
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