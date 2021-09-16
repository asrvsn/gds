import numpy as np
from typing import Tuple, List, Any, Iterable, Callable, Dict, Set
import shortuuid
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import random
from functools import reduce
from inspect import signature
from itertools import chain
import datetime
import pdb

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
		if dict.__contains__(self, key):
			return dict.__getitem__(self, key)
		else:
			return dict.__getitem__(self, (key[1], key[0]))

	def __contains__(self, key: Tuple[Any, Any]):
		return dict.__contains__(self, key) or dict.__contains__(self, (key[1], key[0]))

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
	xi, yi, vals = [], [], []
	m, n = 0, 0
	for r, x in enumerate(X):
		m += 1
		for c, y in enumerate(Y):
			if r == 0:
				n += 1
			v = fun(x, y)
			if v != None and v != 0:
				xi.append(r)
				yi.append(c)
				vals.append(v)
	data = coo_matrix((vals, (xi, yi)), shape=(m, n))
	return data

def oneof(xs: List[bool]):
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

def now():
	return datetime.datetime.now()

def disambiguate_strings(ls: List[str]):
	ret = []
	mp = dict()
	for s in ls:
		if s in mp: 
			mp[s] += 1
			ret.append(s + f'_{mp[s]}')
		else:
			mp[s] = 1
			ret.append(s)
	return ret
