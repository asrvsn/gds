import networkx as nx
import numpy as np
import scipy.sparse as sp

def degree_matrix(G):
	A = nx.to_scipy_sparse_matrix(G, format="csr")
	n, m = A.shape
	diags = A.sum(axis=1)
	D = sp.spdiags(diags.flatten(), [0], m, n, format="csr")
	return D