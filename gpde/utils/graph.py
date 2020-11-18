''' Graph utilities ''' 
import networkx as nx

def grid_graph_with_pos(m: int, n: int, **kwargs) -> nx.Graph:
	G = nx.grid_2d_graph(n, m, **kwargs)
	# TODO add pos
	return G

def get_lattice_boundary(G: nx.Graph) -> (nx.Graph, nx.Graph, nx.Graph, nx.Graph, nx.Graph):
	''' Get boundary of lattice graph, i.e. graph with tuple-valued nodes ''' 
	# TODO: doesn't work for hexagonal lattice.
	nodes = set(G.nodes())
	edges = set(G.edges())
	xmin, xmax = min(map(lambda a: a[0], nodes)), max(map(lambda a: a[0], nodes))
	ymin, ymax = min(map(lambda a: a[1], nodes)), max(map(lambda a: a[1], nodes))
	dG, dG_L, dG_R, dG_T, dG_B = nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()
	for n in nodes:
		if n[0] == xmin:
			dG_L.add_node(n)
			dG.add_node(n)
		if n[0] == xmax:
			dG_R.add_node(n)
			dG.add_node(n)
		if n[1] == ymin:
			dG_T.add_node(n)
			dG.add_node(n)
		if n[1] == ymax:
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

# def no_slip(G: nx.Graph) -> Callable:
# 	''' Create no-slip velocity condition along x boundaries of grid graph ''' 
# 	boundary = set()
# 	nodes = set(G.nodes())
# 	for node in nodes:
# 		if G.degree(node) < 4:
# 			N = (node[0], node[1] - 1)
# 			S = (node[0], node[1] + 1)
# 			E = (node[0]-1, node[1])
# 			W = (node[0]+1, node[1])
# 			# Add condition to normal of missing nodes
# 			if not (N in nodes and S in nodes):
# 				boundary.add((node, E))
# 				boundary.add((node, W))
# 			if not (E in nodes and W in nodes):
# 				boundary.add((node, N))
# 				boundary.add((node, S))
# 	def bc(t, e):
# 		if e in boundary or (e[1], e[0]) in boundary:
# 			return 0.
# 		return None
# 	return bc