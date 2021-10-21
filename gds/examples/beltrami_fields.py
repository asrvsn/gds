'''
Computation of discrete Beltrami fields
'''

import networkx as nx
import numpy as np
import pdb
import colorcet as cc
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
import cvxpy as cp

import gds
from gds.types import *
from .fluid_projected import *

def beltrami(G: nx.Graph):
	flow = gds.edge_gds(G)
	y0 = np.random.uniform(size=flow.ndim)
	y0 = flow.leray_project(y0)
	flow.set_initial(y0=lambda x: y0[flow.X[x]])
	flow.set_evolution()