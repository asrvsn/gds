class velocity_eq(edge_pde):
	''' Incompressible Navier-Stokes velocity equation ''' 

	def __init__(self, G: nx.Graph, pressure: vertex_pde, kinematic_viscosity: float=1.0):
		def f(t, self):
			return -self.advect_self() - pressure.grad() + kinematic_viscosity * self.helmholtzian()
		super().__init__(G, f)

		# Track metrics for turbulence transition
		self.metrics = {
			'n': 0,
			'welf_mu': None,
			'welf_M2': None,
			'sigma': None,
		}

	def step(self, dt: float):
		super().step(dt)
		if self.metrics['n'] == 0:
			self.metrics['welf_mu'] = self.y
			self.metrics['welf_M2'] = np.zeros_like(self.y)
			self.metrics['sigma'] = 0.
		else:
			n = self.metrics['n'] + 1
			new_mu = self.metrics['welf_mu'] + (self.y - self.metrics['welf_mu']) / n
			self.metrics['welf_M2'] += (self.y - self.metrics['welf_mu']) * (self.y - new_mu)
			self.metrics['sigma'] = np.sqrt((self.metrics['welf_M2'] / n).sum())
			self.metrics['welf_mu'] = new_mu
		self.metrics['n'] += 1
		print('velocity sigma:', self.metrics['sigma'])