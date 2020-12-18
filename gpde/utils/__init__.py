import numpy as np
from typing import Tuple, List, Any, Iterable, Callable, Dict, Set
from scipy.integrate import RK45
import shortuuid
from scipy.sparse import csr_matrix, coo_matrix
import random

def set_seed(seed=None):
	random.seed(seed)
	np.random.seed(seed)

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
			return dict.__getitem__(self, key)
		else:
			return dict.__getitem__(self, (key[1], key[0]))

def replace(arr: np.ndarray, replace_at: list, replace_with: np.ndarray):
	arr[replace_at] = replace_with
	return arr

def subclass(cls, attrs: Dict[str, Any]):
	subclass_name = cls.__name__ + '_' + shortuuid.uuid()
	return type(subclass_name, (cls,), attrs)

def attach_dyn_props(instance, props: Dict[str, Callable]):
    """Attach prop_fn to instance with name prop_name.
    Assumes that prop_fn takes self as an argument.
    Reference: https://stackoverflow.com/a/1355444/509706
    """
    attrs = {k: property(v) for k, v in props.items()}
    instance.__class__ = subclass(instance.__class__, attrs)


def dict_fun(m: Dict) -> Callable: 
	return lambda x: m[x] if x in m else None

def sparse_coo(X: Iterable[Any], Y: Iterable[Any], fun: Callable[[Any, Any], float]) -> coo_matrix:
	data = coo_matrix((len(X), len(Y)))
	for r, x in enumerate(X):
		for c, y in enumerate(Y):
			data[r, c] = fun(x, y)
	return data
