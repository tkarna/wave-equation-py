"""
Verify correctness with a stationary geostrophic gyre.
"""
import model
import numpy
import constant
import click


def exact_solution(grid, t):
    """
    Exact solution for a stationary geostrophic gyre.
    """
    g = constant.g
    f = 10.0  # TODO include in constant
    amp = 0.02
    x0 = 0.0
    y0 = 0.0
    sigma = 0.4
    x = grid.x_t_2d - x0
    y = grid.y_t_2d - y0
    x_u = grid.x_u_2d - x0
    y_u = grid.y_u_2d - y0
    x_v = grid.x_v_2d - x0
    y_v = grid.y_v_2d - y0
    elev = amp*numpy.exp(-(x**2+y**2)/sigma**2)
    elev_u = amp*numpy.exp(-(x_u**2+y_u**2)/sigma**2)
    elev_v = amp*numpy.exp(-(x_v**2+y_v**2)/sigma**2)
    u = g/f*2*y_u/sigma**2*elev_u
    v = -g/f*2*x_v/sigma**2*elev_v
    return u, v, elev


def initial_solution(grid):
    """Set initial condition for water elevation"""
    return exact_solution(grid, 0)


def bathymetry(grid):
    """Expression for bathymetry"""
    return numpy.ones(grid.T_shape)


@click.command()
@click.option('-b', '--backend', default='numpy', show_default=True,
              type=click.Choice(['numpy', 'ramba', 'numba', 'jax'],
                                case_sensitive=False),
              help='Use given backend.')
@click.option('-n', '--resolution', default=128,
              type=click.IntRange(min=4, max_open=True), show_default=True,
              help='Number of grid cells in x and y direction.')
@click.option('-p', '--runtime-plot', is_flag=True, default=False,
              type=click.BOOL, show_default=True,
              help='Plot solution at runtime.')
def main(**kwargs):
    n = kwargs.pop('resolution')
    model.run(
        n, n,
        initial_solution_func=initial_solution,
        bathymetry_func=bathymetry,
        exact_solution_func=exact_solution,
        vmax=0.02, umax=0.06,
        **kwargs
    )


if __name__ == '__main__':
    main()
