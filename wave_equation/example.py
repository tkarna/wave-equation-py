"""
An example simulation with visualization.
"""
import model
import numpy
import click


def initial_elev(grid):
    """Set initial condition for water elevation"""
    amp = 0.5
    radius = 0.1
    x0 = -0.3
    y0 = -0.2
    dist2 = (grid.x_t_2d - x0)**2 + (grid.y_t_2d - y0)**2
    return amp * numpy.exp(-1.0 * dist2 / radius**2)


@click.command()
@click.option('-b', '--backend', type=click.Choice(['numpy', 'ramba'],
              case_sensitive=False), default='numpy', show_default=True,
              help='Use given backend.')
def main(**kwargs):
    n = 256
    model.run(
        n, n, t_end=1.6,
        initial_elev_func=initial_elev,
        runtime_plot=True, vmax=0.2,
        **kwargs
    )


if __name__ == '__main__':
    main()
