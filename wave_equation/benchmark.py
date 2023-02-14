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
def main(**kwargs):
    backend_list = kwargs.pop('backend')
    # numpy needs to run before numba
    ordered_list = ['numpy', 'numba', 'ramba', 'jax']
    backend_list = [b for b in ordered_list if b in backend_list]
    ntimestep = 200
    dt = 1e-4
    t_export = dt*50
    reso_list = [128, 256, 512, 1024, 1536]

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
            timings[(b, r)] = t/ntimestep

    # compare against numpy
    speedup = {}
    for b in backend_list:
        for r in reso_list:
            t = timings[(b, r)]
            t_ref = timings[('numpy', r)]
            speedup[(b, r)] = t_ref/t

    dofs_list = [r*r + 2 * r*(r+1) for r in reso_list]

    # print table
    backend_cols = ' '.join(f'{b:7s}' for b in backend_list)
    print(f'size dofs     {backend_cols}')
    for r, dofs in zip(reso_list, dofs_list):
        time_cols = ' '.join(f'{timings[(b, r)]:7.2e}'
                             for b in backend_list)
        print(f'{r:4d} {dofs:8d} {time_cols}')

    # make a plot
    import matplotlib.pyplot as plt
    fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    fig.suptitle('Linear wave equation')
    ax = ax_list[0]
    x = dofs_list
    for b in backend_list:
        y = [timings[(b, r)] for r in reso_list]
        ax.loglog(x, y, label=b, marker='o')
        if b == 'numpy':
            for xi, yi, ri in zip(x, y, reso_list):
                ax.text(xi, yi, f'{ri}x{ri}',
                        va='top', ha='left', size='small')
    ax.set_xlabel('Problem size (degrees of freedom)')
    ax.set_ylabel('Time per time step (s)')
    ax.grid(True, which='both')
    ax.legend()

    ax = ax_list[1]
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