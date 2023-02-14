"""
Verify correctness with standing wave test case.
"""
import model
import numpy
import math
import constant
import click


def exact_elev(grid, t):
    """
    Exact solution for elevation field.

    Returns time-dependent elevation of a 2D standing wave in a rectangular
    domain.
    """
    amp = 0.5
    c = math.sqrt(constant.g * constant.h)
    n = 1
    sol_x = numpy.sin(2 * n * numpy.pi * grid.x_t_2d / grid.lx)
    m = 1
    sol_y = numpy.sin(2 * m * numpy.pi * grid.y_t_2d / grid.ly)
    omega = c * numpy.pi * math.sqrt((n/grid.lx)**2 + (m/grid.ly)**2)
    sol_t = numpy.cos(2 * omega * t)
    return amp * sol_x * sol_y * sol_t


def initial_elev(grid):
    """Set initial condition for water elevation"""
    return exact_elev(grid, 0)


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
        initial_elev_func=initial_elev,
        exact_elev_func=exact_elev,
        **kwargs
    )


if __name__ == '__main__':
    main()
