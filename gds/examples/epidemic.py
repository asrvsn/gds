
import numpy as np
import networkx as nx
import gds
import json
from pathlib import Path
import random
import colorcet as cc

def SIR_model(G, 
		dS=0.1, dI=0.5, dR=0.0, 	# Diffusive terms
		muS=0.1, muI=0.3, muR=0.1, 	# Death rates
		Lambda=0.5, 				# Birth rate 
		beta=0.2, 					# Rate of contact
		gamma=0.2, 					# Rate of recovery
		initial_population=100,
		patient_zero=None,
		**kwargs):
	''' 
	Reaction-Diffusion SIR model
	Based on Huang et al, https://www.researchgate.net/publication/281739911_The_reaction-diffusion_system_for_an_SIR_epidemic_model_with_a_free_boundary
	'''
	susceptible = gds.node_gds(G, **kwargs)
	infected = gds.node_gds(G, **kwargs)
	recovered = gds.node_gds(G, **kwargs)

	susceptible.set_evolution(dydt=lambda t, y:
		dS*susceptible.laplacian() - muS*susceptible.y - beta*susceptible.y*infected.y + Lambda
	)

	infected.set_evolution(dydt=lambda t, y:
		dI*infected.laplacian() + beta*susceptible.y*infected.y - muI*infected.y - gamma*infected.y
	)

	recovered.set_evolution(dydt=lambda t, y:
		dR*recovered.laplacian() + gamma*infected.y - muR*recovered.y
	)

	if patient_zero is None:
		patient_zero = random.choice(list(G.nodes()))
	print(patient_zero)
	susceptible.set_initial(y0=lambda x: initial_population)
	infected.set_initial(y0=lambda x: 1 if x == patient_zero else 0.)

	sys = gds.couple({
		'Susceptible': susceptible,
		'Infected': infected,
		'Recovered': recovered
	})

	return sys

def US_cities(k=500):
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
			if i != j and dist(ix, jx) <= 7.5:
				edges.append((i, j))
				key = (i, j) if i < j else (j, i)
				weights[key] = min(1/dist(ix, jx), 100.)
	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	nx.set_node_attributes(G, dict(zip(nodes, map(get_pos, nodes))), 'pos')
	nx.set_edge_attributes(G, weights, 'w')
	return G

if __name__ == '__main__':
	random.seed(0)
	G = US_cities()
	sys = SIR_model(G, beta=0.4, muS=0., muI=0., muR=0., Lambda=0., w_key='w', patient_zero=233)
	gds.render(sys, title='Diffusive SIR model on a network', x_rng=(-1.05, -0.35), y_rng=(0.6, 1.1), node_size=0.015, node_rng=(0, 100))

