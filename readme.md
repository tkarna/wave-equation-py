# Linear wave equation benchmark

Python implementation of the linear wave equation on structured Arakawa C grid.

## Equations

Solves the linear wave equations for water elevation, $\eta$, and depth-averaged flow velocity, $\bar{u} = [u, v]$, where
$u$ and $v$ are the $x$ and $y$ components, respectively.

$$\begin{eqnarray}
\frac{\partial\eta}{\partial t} + h \nabla\cdot(\bar{u}) &=& 0 \\
\frac{\partial\bar{u}}{\partial t} + g \nabla(\eta) &=& 0
\end{eqnarray}$$

where $h$ and $g$ stand for the water depth and gravitational acceleration, respectively.

## Discretization

The equations are solved on the Arakawa C grid. Given a structured 2D grid, the water elevation is defined at the cell centers (T point). The $u$ and $v$ velocities are defined at the center of the edges (U and V points).

The equations are marched in time with the explicit 3-stage, 3rd order Strong Stability Preserving Runge-Kutta method, SSPRK(3,3).

## Backends

Currently `numpy`, `numba`, `ramba`, and `jax` backends are supported. Choose with the `-b` commandline argument. To use jax on GPUs, use `jax-gpu` backend.

## Example

Run an example simulation and visualize the elevation field.

```python
python example.py
python example.py -b numba -n 512
```

Choose the backend and grid resolution with the `-b`  and `-n` options.

## Test

Run a standing wave test case and compare the result against the exact solution.

```python
python test.py -n 128
```

To visualize the solution add `-p` option.

Run a convergence test with a suite of different grid resolutions

```python
python convergence.py
```

## Benchmark

Run the standing wave test case with different backends and compare run times.

```python
python benchmark.py
python benchmark.py -p -b numpy -b numba -n 128 -n 256 -n 512 -n 1024
```

Generate a comparison plot with `-p` option.
