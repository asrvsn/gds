import numpy as np
from typing import Any, Union, Tuple, Callable, NewType, Iterable, Dict
import pdb
import os.path
import os
import hickle as hkl
import cloudpickle
from tqdm import tqdm

from .types import *

''' System objects ''' 

class System:
	def __init__(self, stepper: Steppable, observables: Dict[str, Observable]):
		self._stepper = stepper
		self._observables = observables

	@property
	def stepper(self):
		return self._stepper

	@property 
	def observables(self) -> Dict[str, Observable]:
		return self._observables

	def solve_to_disk(self, T: float, dt: float, folder: str, parent='runs'): 
		assert os.path.isdir(parent), f'Parent directory "{parent}" does not exist'
		path = parent + '/' + folder
		if not os.path.isdir(path):
			os.mkdir(path)
		dump = dict()
		obs_items = self.observables.items()
		for name, obs in obs_items:
			dump[name] = []
		t = 0.
		try:
			with tqdm(total=int(T / dt), desc=folder) as pbar:
				while t < T:
					self.stepper.step(dt)
					for name, obs in obs_items:
						dump[name].append(obs.y.copy())
					t += dt
					pbar.update(1)
		finally:
			# Dump simulation data
			for name, data in dump.items():
				hkl.dump(np.array(data), f'{path}/{name}.hkl', mode='w', compression='gzip')
			# Dump system object
			with open(f'{path}/system.pkl', 'wb') as f:
				self.dt = dt # Save the dt (hacky)
				cloudpickle.dump(self, f)

	@staticmethod
	def from_disk(folder: str, parent='runs'):
		path = parent + '/' + folder
		assert os.path.isdir(path), 'The given path does not exist'
		with open(f'{path}/system.pkl', 'rb') as f:
			sys = cloudpickle.load(f)
		data = dict()
		n = 0
		for name in sys.observables.keys():
			data[name] = hkl.load(f'{path}/{name}.hkl')
			n = data[name].shape[0]
		sys_dt = sys.dt

		class DummySteppable(Steppable):
			def __init__(self):
				self.t = 0.
				self.i = 0

			def step(self, dt: float):
				T = self.t + dt
				while self.t < T and self.i < n:
					self.t += sys_dt
					self.i += 1

			def reset(self):
				self.t = 0.
				self.i = 0

		stepper = DummySteppable()
		for name, obs in sys.observables.items():
			obs.history = data[name] # Hacky
			attach_dyn_props(obs, {'y': lambda self: self.history[stepper.i], 't': lambda _: stepper.t})

		return System(stepper, sys.observables)
