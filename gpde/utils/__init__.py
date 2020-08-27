import numpy as np
from typing import Tuple, List, Any, Iterable, Callable, Dict
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

def replace(arr: np.ndarray, replace_at: list, replace_with: np.ndarray):
	arr[replace_at] = replace_with
	return arr

def attach_dyn_props(instance, props: Dict[str, Callable]):
    """Attach prop_fn to instance with name prop_name.
    Assumes that prop_fn takes self as an argument.
    Reference: https://stackoverflow.com/a/1355444/509706
    """
    class_name = instance.__class__.__name__ + 'Child'
    props = {k: property(v) for k, v in props.items()}
    child_class = type(class_name, (instance.__class__,), props)

    instance.__class__ = child_class