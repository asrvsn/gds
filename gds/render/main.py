''' Rendering utilities ''' 

from gds.types import *
from gds.system import *
from .base import *
from .bokeh import *

def render(obj: Union[Observable, System], **kwargs):
	if isinstance(obj, Observable):
		sys = System(obj, {'observable': obj})
		LiveRenderer(sys, **kwargs).start()
	elif isinstance(obj, System):
		LiveRenderer(obj, **kwargs).start()
	else:
		raise Exception('Please pass an Observable or System for rendering')