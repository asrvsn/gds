''' Dynamics on faces ''' 

import pdb
import gds
import networkx as nx
import numpy as np

def test_orientation():
	G = gds.hexagonal_lattice(3, 4)
	# G = gds.random_planar_graph(100, 0.2)

	eq1 = gds.face_gds(G)
	eq1.set_evolution(nil=True)
	eq1.set_initial(y0=lambda x: 0.5)
	eq2 = gds.node_gds(G)
	eq2.set_evolution(nil=True)
	sys = gds.couple({
		'eq1': eq1,
		'eq2': eq2,
	})
	gds.render(sys, face_orientations=True)

def k_torus_faces():
	G = gds.k_torus(2)
	print('faces:', len(G.faces))
	obs = gds.face_gds(G)
	obs.set_evolution(nil=True)
	gds.render(obs)

if __name__ == '__main__':
	# test_orientation()
	k_torus_faces()

