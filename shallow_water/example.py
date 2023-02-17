"""
An example simulation with visualization.
"""
import model
import numpy
import click


def initial_solution(grid):
    """Set initial condition for u, v, elevation"""
    amp = 0.5
    radius = 0.3
    x0 = 0.0
    y0 = 0.0
    dist2 = (grid.x_t_2d - x0)**2 + (grid.y_t_2d - y0)**2
    elev = amp * numpy.exp(-1.0 * dist2 / radius**2)
    u = numpy.zeros(grid.U_shape, dtype=elev.dtype)
    v = numpy.zeros(grid.V_shape, dtype=elev.dtype)
    return u, v, elev


def bathymetry(grid):
    """Expression for bathymetry"""
    return 1.0 + numpy.exp(-(grid.x_t_2d**2/0.4))


@click.command()
@click.option('-b', '--backend', default='numpy', show_default=True,
              type=click.Choice(['numpy', 'ramba', 'numba', 'jax'],
                                case_sensitive=False),
              help='Use given backend.')
@click.option('-n', '--resolution', default=128,
              type=click.IntRange(min=4, max_open=True), show_default=True,
              help='Number of grid cells in x and y direction.')
def main(**kwargs):
    n = kwargs.pop('resolution')
    model.run(
        n, n,
        initial_solution_func=initial_solution,
        bathymetry_func=bathymetry,
        t_end=1.0,
        runtime_plot=True, plot_energy=True,
        vmax=0.2,
        **kwargs
    )


if __name__ == '__main__':
    main()
