'''
Synchronization phenomena in networked dynamics
'''

def kuramoto(G, k: float, w: Callable):
	'''
	Kuramoto lattice
	k: coupling gain
	w: intrinsic frequency
	'''
	phase = gds.node_gds(G)
	w_vec = np.array([w(x) for x in phase.X])
	phase.set_evolution(dydt=lambda t, y: w_vec + k * phase.div(np.sin(phase.grad())))
	amplitude = phase.project(GraphDomain.nodes, lambda t, y: np.sin(y))
	return phase, amplitude

