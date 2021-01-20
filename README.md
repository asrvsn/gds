<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">gds: graph dynamical systems library</h3>

  <p align="center">
    Simulate discrete- and continuous-time dynamics on networks, including PDEs, coupled map lattices, generalized cellular automata, and more.
    <br />
    <a href="https://a0s.co/docs/gds"><strong>Explore the docs [coming soon]»</strong></a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about">About</a>
      <ul>
        <li><a href="#features">Features</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
     <li><a href="#gallery">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About

![Product Name Screen Shot][product-screenshot]


### Features

* Built directly on top of [NetworkX](https://networkx.org/); represent observables (currently) on vertices or edges 
* Support for time-varying [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_boundary_condition) and [Neumann](https://en.wikipedia.org/wiki/Neumann_boundary_condition) boundary conditions 
* Support for [kinematic equations](https://en.wikipedia.org/wiki/Kinematics) of motion via [CVXPY](https://www.cvxpy.org/)
* Couple multiple discrete- or continuous-time observables across distinct graphs
* Automatic calculation of Lyapunov spectra using [PyTorch](https://pytorch.org/) and [jax](https://github.com/google/jax)

As well as in-browser interactive rendering with [Bokeh](https://bokeh.org/).


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* `conda` (recommended) or python >= 3.7 

This has not been tested in all environments (especially Windows), so please report bugs.

### Installation

* In local environment
```sh
pip install git+git://github.com/asrvsn/gds.git
```
* As a project dependency
```sh
# Add to `requirements.txt`
-e git://github.com/asrvsn/gds.git#egg=gds
```


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation (coming soon)](https://a0s.co/docs/gds)_

### Example: Heat equation with time-varying boundary

Definition:
```python
G = gds.grid_graph(10, 10)
temperature = gds.node_gds(G)
temperature.set_evolution(dydt=lambda t, y: temperature.laplacian())
temperature.set_constraints(dirichlet=gds.combine_bcs([
	lambda t, x: 0 if x[0] == 0 else None,
	lambda t, x: np.sin(t+x[1]/4)**2 if x[0] == 9 else None
]))
gds.render(temperature)
```

Result:

### Example: SIR epidemic on a network

Definition:
```python

```

Result:

### Example: Rayleigh-Benard convection

Definition:
```python
G = gds.grid_graph(10, 10)
temperature, velocity_x, velocity_y = gds.node_gds(G), gds.node_gds(G), gds.node_gds(G)

```

Result:


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).

* Detailed docs
* Lyapunov exponent calculation for continuous-time systems (requiring use of differentiable integrators)
* Probabilistic cellular automata
* Observables on faces, tets, and higher-dimensional objects


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- ACKNOWLEDGEMENTS -->
## References

* []()
* []()
* []()
