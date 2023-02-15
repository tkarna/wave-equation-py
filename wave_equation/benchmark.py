"""
Run a suite of resolutions with different backends.
"""
import model
import click
from test import initial_elev, exact_elev


@click.command()
@click.option('-b', '--backend', default=['numpy', 'ramba', 'numba', 'jax'],
              multiple=True, show_default=True,
              type=click.Choice(['numpy', 'ramba', 'numba', 'jax'],
                                case_sensitive=False),
              help='Use given backend.')
@click.option('-n', '--resolution', default=[128, 256, 512, 1024],
              multiple=True,
              type=click.IntRange(min=4, max_open=True), show_default=True,
              help='Number of grid cells in x and y direction.')
@click.option('-p', '--generate-plot', is_flag=True, default=False,
              type=click.BOOL, show_default=True,
              help='Generate a plot of timings.')
def main(**kwargs):
    backend_list = kwargs['backend']
    generate_plot = kwargs['generate_plot']
    # numpy needs to run before numba
    ordered_list = ['numpy', 'numba', 'ramba', 'jax']
    backend_list = [b for b in ordered_list if b in backend_list]
    ntimestep = 200
    dt = 1e-4
    t_export = dt*50
    reso_list = kwargs['resolution']

    # run models
    timings = {}
    for b in backend_list:
        for r in reso_list:
            t, e = model.run(
                r, r,
                initial_elev_func=initial_elev,
                exact_elev_func=exact_elev,
                dt=dt, ntimestep=ntimestep, t_export=t_export,
                backend=b,
            )
            timings[(b, r)] = t/ntimestep*1000.

    dofs_list = [r*r + 2 * r*(r+1) for r in reso_list]

    # print table
    backend_cols = ' '.join(f'{b:8s}' for b in backend_list)
    print(f'size dofs     {backend_cols}')
    for r, dofs in zip(reso_list, dofs_list):
        time_cols = ' '.join(f'{timings[(b, r)]:8.4g}'
                             for b in backend_list)
        print(f'{r:4d} {dofs:8d} {time_cols}')

    if generate_plot:
        # make a plot
        import matplotlib.pyplot as plt

        plot_speedup = 'numpy' in backend_list
        if plot_speedup:
            # compare against numpy
            speedup = {}
            for b in backend_list:
                for r in reso_list:
                    t = timings[(b, r)]
                    t_ref = timings[('numpy', r)]
                    speedup[(b, r)] = t_ref/t
        nplots = 2 if plot_speedup else 1
        fig, ax_list = plt.subplots(nrows=1, ncols=nplots,
                                    figsize=(6.5*nplots, 5),
                                    squeeze=False)

        fig.suptitle('Linear wave equation')
        ax = ax_list[0, 0]
        x = dofs_list
        for b in backend_list:
            y = [timings[(b, r)]/1000. for r in reso_list]
            ax.loglog(x, y, label=b, marker='o')
        for xi, ri in zip(x, reso_list):
            t_min = min([timings[(b, ri)]/1000. for b in backend_list])
            ax.text(xi, t_min, f'{ri}x{ri}',
                    va='top', ha='left', size='small')
        ax.set_xlabel('Problem size (degrees of freedom)')
        ax.set_ylabel('Time per time step (s)')
        ax.grid(True, which='both')
        ax.legend()

        if plot_speedup:
            ax = ax_list[0, 1]
            for b in backend_list:
                y = [speedup[(b, r)] for r in reso_list]
                ax.semilogx(x, y, label=b, marker='o')
            ax.set_xlabel('Problem size (degrees of freedom)')
            ax.set_ylabel('Speed-up vs. numpy')
            ax.grid(True, which='both')
            ax.legend()

        # save figure
        b_str = '-'.join(backend_list)
        r_str = '-'.join(map(str, reso_list))
        img_name = f'waveeq_{b_str}_{r_str}.png'
        print(f'Saving {img_name}')
        fig.savefig(img_name, bbox_inches='tight', dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    main()
