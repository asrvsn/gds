from typing import Tuple, List, Any
from collections.abc import Iterable
from scipy.integrate import RK45

def destructure(xs: Iterable[Any]) -> List[Any]:
	ret = []
	def helper(sub):
		for x in sub:
			if isinstance(x, Iterable):
				helper(x)
			else:
				ret.append(x)
	helper(xs)
	return ret

class bidict(dict):
	''' dictionary for bi-directional keys ''' 
	def __getitem__(self, key: Tuple[Any, Any]):
		if key in self:
			return dict.__getitem__(key)
		else:
			return dict.__getitem__((key[1], key[0]))

class CoupledRK45(RK45):
	def __init__(self, *xs: Tuple[RK45]):
		''' Couple multiple integrators in time ''' 
		assert all([x.t == xs[0].t for x in xs]), 'Cannot couple integrators at different times'
		self.xs = list(xs)
