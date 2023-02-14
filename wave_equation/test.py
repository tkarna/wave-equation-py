"""
Verify correctness with standing wave test case.
"""
import model
import numpy
import math
import constant


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


if __name__ == '__main__':
    n = 128
    model.run(
        n, n,
        initial_elev_func=initial_elev,
        exact_elev_func=exact_elev,
        runtime_plot=False
    )
