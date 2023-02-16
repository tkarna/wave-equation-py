"""
An example simulation with visualization.
"""
import model
import numpy
import click


def initial_elev(x, y):
    """Set initial condition for water elevation"""
    amp = 0.5
    radius = 0.3
    x0 = 0.0
    y0 = 0.0
    dist2 = (x - x0)**2 + (y - y0)**2
    return amp * numpy.exp(-1.0 * dist2 / radius**2)


def bathymetry(x, y):
    """Expression for bathymetry"""
    return 1.0 + numpy.exp(-(x**2/0.4))


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
        initial_elev_func=initial_elev,
        bathymetry_func=bathymetry,
        t_end=1.0,
        runtime_plot=True, plot_energy=True,
        vmax=0.2,
        **kwargs
    )


if __name__ == '__main__':
    main()
