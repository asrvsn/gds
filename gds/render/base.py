from typing import List, Tuple, Any, Dict, NewType

from gds.types import Observable

''' Common types ''' 

Canvas = List[List[List[List[Observable]]]]
Plot = Any
PlotID = NewType('PlotID', str)

''' Layout creators ''' 

def single_canvas(item: Any) -> Canvas:
	''' Render all item in the same plot ''' 
	if isinstance(item, Iterable):
		return [[[list(item)]]]
	else:
		return [[[[item]]]]

def grid_canvas(observables: List[Observable], ncols: int=2) -> Canvas:
	''' Render all observables separately as items on a grid ''' 
	canvas = []
	for i, obs in enumerate(observables):
		if i % ncols == 0:
			canvas.append([])
		row = canvas[-1]
		row.append([(obs,)])
	return canvas