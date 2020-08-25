''' Rendering utilities ''' 

from gpde.core import *
from .bokeh import *

def render_mpl(sys: System):
	''' Render video as matplotlib frames ''' 
	raise NotImplementedError

def render_bokeh(renderer: Renderer):
	''' Render as a Bokeh web app ''' 
	pass

__all__ = (
	'render_mpl',
	'render_bokeh'
)