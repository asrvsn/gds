'''
Implicit midpoint method adapted from John Burkardt's code: https://people.sc.fsu.edu/~jburkardt/py_src/midpoint/midpoint.html
Mirrors API of `scipy.integrate`
'''

from typing import Callable
from scipy.optimize import fsolve
import numpy as np

class ImplicitMidpoint:
  def __init__(self, fun: Callable, t0: float, y0: np.ndarray, t_bound: float, max_step: float=1e-3):
    self.t = t0
    self.y = y0
    self.fun = fun
    self.t_bound = t_bound
    self.status = 'running' if t0 < t_bound else 'finished'
    self.max_step = max_step
    self.n = y0.size

  def step(self):
    if self.status != 'running':
      raise RuntimeError('Attempt to step on a failed or finished solver.')
    if self.n == 0 or self.t == self.t_bound:
      self.status = 'finished'
    else:
      dt = min(self.max_step, self.t_bound - self.t)
      th = self.t + 0.5 * dt
      yh = self.y + 0.5 * dt * self.fun(self.t, self.y)
      yh = fsolve(midpoint_residual, yh, args=(self.fun, self.t, self.y, th))

      self.t += dt
      self.y = 2.0 * yh - self.y


def midpoint ( f, tspan, y0, n ):

#*****************************************************************************80
#
## midpoint() uses the implicit midpoint method to solve an ODE.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    07 April 2021
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    function f: evaluates the right hand side of the ODE.  
#
#    real tspan[2]: the starting and ending times.
#
#    real y0[m]: the initial conditions. 
#
#    integer n: the number of steps.
#
#  Output:
#
#    real t[n+1], y[n+1,m]: the solution estimates.
#

  if ( np.ndim ( y0 ) == 0 ):
    m = 1
  else:
    m = len ( y0 )

  t = np.zeros ( n + 1 )
  y = np.zeros ( [ n + 1, m ] )

  dt = ( tspan[1] - tspan[0] ) / float ( n )

  t[0] = tspan[0];
  y[0,:] = y0

  for i in range ( 0, n ):

    to = t[i]
    yo = y[i,:]

    th = to + 0.5 * dt
    yh = yo + 0.5 * dt * f ( to, yo )
    yh = fsolve ( midpoint_residual, yh, args = ( f, to, yo, th ) )

    tp = to + dt
    yp = 2.0 * yh - yo

    t[i+1]   = tp
    y[i+1,:] = yp

  return t, y

def midpoint_residual ( yh, f, to, yo, th ):

#*****************************************************************************80
#
## midpoint_residual() evaluates the midpoint residual.
#
#  Discussion:
#
#    We are seeking a value YH defined by the implicit equation:
#
#      YH = YO + ( TH - TO ) * F ( TH, YH )
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    21 October 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real yh: the estimated solution value at the midpoint time.
#
#    function f: evaluates the right hand side of the ODE.  
#
#    real to, yo: the old time and solution value.
#
#    real th: the midpoint time.
#
#  Output:
#
#    real value: the midpoint residual.
#
  value = yh - yo - ( th - to ) * f ( th, yh );

  return value

def midpoint_test ( ):

#*****************************************************************************80
#
## midpoint_test() tests midpoint().
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    21 October 2020
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'midpoint_test:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test midpoint() for solving an ordinary differential equation.' )

  tspan = np.array ( [ 0.0, 2.0 ] )
  y0 = 5.1765
  n = 100
  midpoint_humps_test ( tspan, y0, n )

  tspan = np.array ( [ 0.0, 10.0 ] )
  y0 = np.array ( [ 5000, 100 ] )
  n = 100
  midpoint_predator_prey_test ( tspan, y0, n )
#
#  Terminate.
#
  print ( '' )
  print ( 'midpoint_test:' )
  print ( '  Normal end of execution.' )
  return

def midpoint_humps_test ( tspan, y0, n ):

#*****************************************************************************80
#
## midpoint_humps_test() solves humps ODE using midpoint().
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    21 October 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real tspan[2]: the time span
#
#    real y0[2]: the initial condition.
#
#    integer n: the number of steps to take.
#
  import matplotlib.pyplot as plt
  import numpy as np

  print ( '' )
  print ( 'midpoint_humps_test:' )
  print ( '  Solve the humps ODE system using midpoint().' )

  t, y = midpoint ( humps_deriv, tspan, y0, n )

  plt.plot ( t, y, 'r-', linewidth = 2 )

  a = tspan[0]
  b = tspan[1]
  if ( a <= 0.0 and 0.0 <= b ):
    plt.plot ( [a,b], [0,0], 'k-', linewidth = 2 )

  ymin = min ( y )
  ymax = max ( y )
  if ( ymin <= 0.0 and 0.0 <= ymax ):
    plt.plot ( [0, 0], [ymin,ymax], 'k-', linewidth = 2 )

  plt.grid ( True )
  plt.xlabel ( '<--- T --->' )
  plt.ylabel ( '<--- Y(T) --->' )
  plt.title ( 'humps ODE solved by midpoint()' )

  filename = 'midpoint_humps.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.show ( block = False )
  plt.close ( )

  return

def midpoint_predator_prey_test ( tspan, y0, n ):

#*****************************************************************************80
#
## midpoint_predator_prey_test() solves predator prey using midpoint().
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    21 October 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real tspan[2]: the time span
#
#    real y0[2]: the initial condition.
#
#    integer n: the number of steps to take.
#
  import matplotlib.pyplot as plt
  import numpy as np

  print ( '' )
  print ( 'midpoint_predator_prey_test:' )
  print ( '  Solve the predator prey ODE using midpoint().' )

  t, y = midpoint ( predator_prey_deriv, tspan, y0, n )

  plt.plot ( y[:,0], y[:,1], 'r-', linewidth = 2 )

  plt.grid ( True )
  plt.xlabel ( '<--- Prey --->' )
  plt.ylabel ( '<--- Predators --->' )
  plt.title ( 'predator prey ODE solved by midpoint()' )

  filename = 'midpoint_predator_prey.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.show ( block = False )
  plt.close ( )

  return

def humps_deriv ( x, y ):

#*****************************************************************************80
#
## humps_deriv() evaluates the derivative of the humps function.
#
#  Discussion:
#
#    y = 1.0 / ( ( x - 0.3 )**2 + 0.01 ) \
#      + 1.0 / ( ( x - 0.9 )**2 + 0.04 ) \
#      - 6.0
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    real x[:], y[:]: the arguments.
#
#  Output:
#
#    real yp[:]: the value of the derivative at x.
#
  yp = - 2.0 * ( x - 0.3 ) / ( ( x - 0.3 )**2 + 0.01 )**2 \
       - 2.0 * ( x - 0.9 ) / ( ( x - 0.9 )**2 + 0.04 )**2

  return yp

def predator_prey_deriv ( t, rf ):

#*****************************************************************************80
#
## predator_prey_deriv() evaluates the right hand side of the system.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    George Lindfield, John Penny,
#    Numerical Methods Using MATLAB,
#    Second Edition,
#    Prentice Hall, 1999,
#    ISBN: 0-13-012641-1,
#    LC: QA297.P45.
#
#  Input:
#
#    real T, the current time.
#
#    real RF[2], the current solution variables, rabbits and foxes.
#
#  Output:
#
#    real DRFDT[2], the right hand side of the 2 ODE's.
#
  import numpy as np

  r = rf[0]
  f = rf[1]

  drdt =    2.0 * r - 0.001 * r * f
  dfdt = - 10.0 * f + 0.002 * r * f

  drfdt = np.array ( [ drdt, dfdt ] )

  return drfdt

def timestamp ( ):

#*****************************************************************************80
#
## timestamp() prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    21 August 2019
#
#  Author:
#
#    John Burkardt
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return

if ( __name__ == '__main__' ):
  timestamp ( )
  midpoint_test ( )
  timestamp ( )
 
