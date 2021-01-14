import numpy as np
from typing import Tuple, List, Any, Iterable, Callable, Dict, Set
import shortuuid
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import random
from functools import reduce
from inspect import signature
from itertools import chain

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


def dict_fun(m: Dict, def_val: Any=None) -> Callable: 
	return lambda x: m[x] if x in m else def_val

def sparse_product(X: Iterable[Any], Y: Iterable[Any], fun: Callable[[Any, Any], float]) -> coo_matrix:
	data = dok_matrix((len(X), len(Y)))
	for r, x in enumerate(X):
		for c, y in enumerate(Y):
			v = fun(x, y)
			if v is not None:
				data[r, c] = v
	return data.tocoo()

def oneof(xs: List[boolean]):
	return reduce(lambda x, y: x ^ y, xs)

def fun_ary(f: Callable) -> int:
	''' Returns number of arguments required by function ''' 
	return len(signature(f).parameters)

def merge_dicts(xs: Iterable[Dict]) -> Dict:
	ret = dict()
	for x in xs:
		ret.update(x)
	return ret

def flatten(arr: List[Any]) -> List:
	return list(chain.from_iterable(arr))
