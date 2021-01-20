''' Rendering utilities ''' 

from gds.types import *
from gds.system import *
from .base import *
from .bokeh import *

def render(obj: Union[Observable, System], **kwargs):
	if isinstance(obj, Observable):
		sys = System({'observable': obj})
		LiveRenderer(sys, sys.arrange()).start()
	elif isinstance(obj, System):
		LiveRenderer(obj, obj.arrange()).start()
	else:
		raise Exception('Please pass an Observable or System for rendering')