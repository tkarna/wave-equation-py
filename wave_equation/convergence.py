"""
Verify correctness in convergence test.
"""
from test import model, initial_elev, exact_elev
import numpy


def run_convergence(n, dt, refinement_list, show_plot=False):
    dx_list = []
    er_list = []
    for r in refinement_list:
        print(f'Running refinement {r}')
        nr = n * 2**r
        dx = 2.0 / nr
        tr, er = model.run(
            nr, nr,
            initial_elev_func=initial_elev,
            exact_elev_func=exact_elev,
            dt=dt,
            runtime_plot=False
        )
        er_list.append(er)
        dx_list.append(dx)
        print('')
    er_list = numpy.asarray(er_list)
    dx_list = numpy.asarray(dx_list)

    print('r   n         dx     error')
    for r, dx, e, in zip(refinement_list, dx_list, er_list):
        print(f'{r:2d} {n * 2**r:3d} {dx:6.4e} {e:.3e}')

    er_log = numpy.log10(er_list)
    dx_log = numpy.log10(dx_list)

    slope, intercept = numpy.polyfit(dx_log, er_log, 1)
    print(f'Convergence rate: {slope:0.2f}')
    assert slope > 1.8, 'Too low convergence rate: {slope:0.2f}'
    print('PASSED')

    if show_plot:
        import matplotlib.pyplot as plt
        xx = numpy.linspace(dx_list.max()*1.1, dx_list.min() * 0.9, 6)
        yy = 10**(intercept + slope*numpy.log10(xx))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.loglog(dx_list, er_list, marker='+')
        ax.loglog(xx, yy, 'k--', lw=0.5)
        ax.text(xx[-3:-2].mean(), yy[-3:-2].mean(), f'{slope:.2f}', va='top')
        ax.set_xlabel('dx')
        ax.set_ylabel('L2 error')
        ax.grid(True)
        plt.show()


if __name__ == '__main__':
    # base resolution
    n = 8
    dt = 2e-4  # small dt => spatial error dominates
    refinement_list = numpy.arange(0, 4)
    run_convergence(n, dt, refinement_list, show_plot=False)
