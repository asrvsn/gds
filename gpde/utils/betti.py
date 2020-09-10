from typing import List
import itertools
import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix

def betti(G: nx.Graph, verbose = False) -> List[int]:
    ''' 
    Compute the Betti numbers of a topological graph.
    Credit: https://gist.github.com/numpde/16f3a22e352dc43dc01614b50b74645b

    # RA, 2017-11-03, CC-BY-4.0
    
    # Ref: 
    # A. Zomorodian, Computational topology (Notes), 2009
    # http://www.ams.org/meetings/short-courses/zomorodian-notes.pdf
    '''
    
    def DIAGNOSTIC(*params) :
        if verbose : print(*params)
    
    #
    # 1. Prepare maximal cliques
    #
    
    # Compute maximal cliques
    C = nx.find_cliques(G)
    
    # Sort each clique, make sure it's a tuple
    C = [tuple(sorted(c)) for c in C]
    
    DIAGNOSTIC("Number of maximal cliques: {} ({}M)".format(len(C), round(len(C) / 1e6)))
    
    #
    # 2. Enumerate all simplices
    #
    
    # S[k] will hold all k-simplices
    # S[k][s] is the ID of simplex s
    S = []
    for k in range(0, max(len(s) for s in C)) :
        # Get all (k+1)-cliques, i.e. k-simplices, from max cliques mc
        Sk = set(c for mc in C for c in itertools.combinations(mc, k+1))
        # Check that each simplex is in increasing order
        assert(all((list(s) == sorted(s)) for s in Sk))
        # Assign an ID to each simplex, in lexicographic order
        S.append(dict(zip(sorted(Sk), range(0, len(Sk)))))

    for (k, Sk) in enumerate(S) :
        DIAGNOSTIC("Number of {}-simplices: {}".format(k, len(Sk)))
    
    # Euler characteristic
    ec = sum(((-1)**k * len(S[k])) for k in range(0, len(S)))
    
    DIAGNOSTIC("Euler characteristic:", ec)
    
    #
    # 3. Construct the boundary operator
    #
    
    # D[k] is the boundary operator 
    #      from the k complex 
    #      to the k-1 complex
    D = [None for _ in S]

    # D[0] is the zero matrix
    D[0] = lil_matrix((1, G.number_of_nodes()))

    # Construct D[1], D[2], ...
    for k in range(1, len(S)) :
        D[k] = lil_matrix( (len(S[k-1]), len(S[k])) )
        SIGN = np.asmatrix([(-1)**i for i in range(0, k+1)]).transpose()

        for (ks, j) in S[k].items() :
            # Indices of all (k-1)-subsimplices s of the k-simplex ks
            I = [S[k-1][s] for s in sorted(itertools.combinations(ks, k))]
            D[k][I, j] = SIGN
    
    for (k, d) in enumerate(D) : 
        DIAGNOSTIC("D[{}] has shape {}".format(k, d.shape))
    
    # Check that D[k-1] * D[k] is zero
    assert(all((0 == np.dot(D[k-1], D[k]).count_nonzero()) for k in range(1, len(D))))
    
    #
    # 4. Compute rank and dimker of the boundary operators
    #
    
    # Rank and dimker
    rk = [np.linalg.matrix_rank(d.todense()) for d in D]
    ns = [(d.shape[1] - rk[n]) for (n, d) in enumerate(D)]
    
    DIAGNOSTIC("rk:", rk)
    DIAGNOSTIC("ns:", ns)
    
    #
    # 5. Infer the Betti numbers
    #

    # Betti numbers
    # B[0] is the number of connected components
    B = [(n - r) for (n, r) in zip(ns[:-1], rk[1:])]
    
    # Check: Euler-Poincare formula
    assert(ec == sum(((-1)**k * B[k]) for k in range(0, len(B))))
    
    return B