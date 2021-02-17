
import numpy as np
import networkx as nx
import gds
import json
from pathlib import Path
import random

def SIR_model(G, dS=0.1, dI=0.5, dR=0.0, alpha1=0.1, alpha2=0.02, alpha3=0.03, L=0.5, mu=0.1, beta=0.2, r=0.5, **kwargs):
	''' 
	Reaction-Diffusion SIR model
	Based on Lotfi et al, https://www.hindawi.com/journals/ijpde/2014/186437/
	'''
	susceptible = gds.node_gds(G, **kwargs)
	infected = gds.node_gds(G, **kwargs)
	recovered = gds.node_gds(G, **kwargs)

	def N():
		return 1 + alpha1*susceptible.y + alpha2*infected.y + alpha3*susceptible.y*infected.y

	susceptible.set_evolution(dydt=lambda t, y:
		dS*susceptible.laplacian() + L - mu*susceptible.y - beta*susceptible.y*infected.y / N()
	)

	infected.set_evolution(dydt=lambda t, y:
		dI*infected.laplacian() + beta*susceptible.y*infected.y / N()
	)

	recovered.set_evolution(dydt=lambda t, y:
		dR*recovered.laplacian() + r*infected.y - mu*recovered.y
	)

	susceptible.set_initial(y0=lambda x: np.random.uniform())
	infected.set_initial(y0=lambda x: np.random.uniform())

	sys = gds.couple({
		'Susceptible': susceptible,
		'Infected': infected,
		'Recovered': recovered
	})

	return sys

def US_cities(k=200):
	path = str(Path(__file__).parent / 'cities.json')
	cities = json.load(open(path))
	cities = list(filter(lambda c: not (c['state'] in ['Hawaii', 'Alaska']), cities))
	cities = random.choices(cities, k=k)
	nodes = list(range(k))
	weights = dict()
	edges = []
	def get_pos(i):
		return (cities[i]['longitude'], cities[i]['latitude'])
	def dist(x, y):
		return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
	for i in nodes:
		for j in nodes:
			ix, jx = get_pos(i), get_pos(j)
			if i != j and dist(ix, jx) <= 10:
				edges.append((i, j))
				key = (i, j) if i < j else (j, i)
				weights[key] = min(1/dist(ix, jx), 1.)
	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	nx.set_node_attributes(G, dict(zip(nodes, map(get_pos, nodes))), 'pos')
	nx.set_edge_attributes(G, weights, 'w')
	return G

if __name__ == '__main__':
	random.seed(10)
	G = US_cities()
	sys = SIR_model(G, beta=0.01, w_key='w')
	gds.render(sys, title='Diffusive SIR model on a network', x_rng=(-1.05, -0.35), y_rng=(0.6, 1.1), node_size=0.015, dynamic_ranges=True)