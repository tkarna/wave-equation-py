import numpy
import constant
import math
import matplotlib.pyplot as plt
import time as time_mod


def run(grid, initial_elev_func, exact_elev_func=None,
        t_end=1.0, t_export=0.02, dt=None, runtime_plot=False, vmax=0.5):
    """
    Run simulation.
    """
    g = constant.g
    h = constant.h

    # state variables
    elev = numpy.zeros(grid.T_shape, dtype=numpy.float64)
    u = numpy.zeros(grid.U_shape, dtype=numpy.float64)
    v = numpy.zeros(grid.V_shape, dtype=numpy.float64)

    # state for RK stages
    elev1 = numpy.zeros_like(elev)
    elev2 = numpy.zeros_like(elev)
    u1 = numpy.zeros_like(u)
    u2 = numpy.zeros_like(u)
    v1 = numpy.zeros_like(v)
    v2 = numpy.zeros_like(v)

    # tendecies u += dt*dudt
    dudt = numpy.zeros_like(u)
    dvdt = numpy.zeros_like(v)
    delevdt = numpy.zeros_like(elev)

    # initial condition
    elev[...] = initial_elev_func(grid)

    # time step
    if dt is None:
        c = math.sqrt(g*h)
        alpha = 0.5
        dt = alpha * grid.dx / c
        dt = t_export / int(math.ceil(t_export / dt))
    nt = int(math.ceil(t_end / dt))
    print(f'Time step: {dt} s')
    print(f'Total run time: {t_end} s, {nt} time steps')

    def rhs(u, v, elev):
        """
        Evaluate right hand side of the equations
        """
        # terms
        # sign convention: positive on rhs

        # pressure gradient -g grad(elev)
        dudt[1:-1, :] = -g * (elev[1:, :] - elev[:-1, :])/grid.dx
        dvdt[:, 1:-1] = -g * (elev[:, 1:] - elev[:, :-1])/grid.dy

        # periodic boundary
        dudt[0, :] = - g * (elev[0, :] - elev[-1, :])/grid.dx
        dudt[-1, :] = dudt[0, :]
        dvdt[:, 0] = - g * (elev[:, 0] - elev[:, -1])/grid.dy
        dvdt[:, -1] = dvdt[:, 0]

        # velocity divergence -h div(u)
        delevdt[...] = -h * ((u[1:, :] - u[:-1, :])/grid.dx +
                             (v[:, 1:] - v[:, :-1])/grid.dy)

    if runtime_plot:
        plt.ion()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        img = ax.pcolormesh(grid.x_u_1d, grid.y_v_1d, elev.T,
                            vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(img, label='Elevation')
        fig.canvas.draw()
        fig.canvas.flush_events()

    t = 0
    i_export = 0
    next_t_export = 0
    initial_v = None
    tic = time_mod.perf_counter()
    for i in range(nt+1):

        t = i*dt

        if t >= next_t_export:
            elev_max = float(numpy.max(elev))
            u_max = float(numpy.max(u))

            total_v = float(numpy.sum(elev + h)) * grid.dx * grid.dy
            if initial_v is None:
                initial_v = total_v
            diff_v = total_v - initial_v

            print(f'{i_export:2d} {i:4d} {t:.3f} elev={elev_max:7.5f} '
                  f'u={u_max:7.5f} dV={diff_v: 6.3e}')
            if elev_max > 1e3 or not math.isfinite(elev_max):
                print(f'Invalid elevation value: {elev_max}')
                break
            i_export += 1
            next_t_export = i_export * t_export
            if runtime_plot:
                img.update({'array': elev.T})
                fig.canvas.draw()
                fig.canvas.flush_events()

        # SSPRK33 time integrator
        rhs(u, v, elev)
        u1[...] = u + dt*dudt
        v1[...] = v + dt*dvdt
        elev1[...] = elev + dt*delevdt
        rhs(u1, v1, elev1)
        u2[...] = 0.75*u + 0.25*(u1 + dt*dudt)
        v2[...] = 0.75*v + 0.25*(v1 + dt*dvdt)
        elev2[...] = 0.75*elev + 0.25*(elev1 + dt*delevdt)
        rhs(u2, v2, elev2)
        u[...] = u/3 + 2/3*(u2 + dt*dudt)
        v[...] = v/3 + 2/3*(v2 + dt*dvdt)
        elev[...] = elev/3 + 2/3*(elev2 + dt*delevdt)

    duration = time_mod.perf_counter() - tic
    print(f'Duration: {duration:.2f} s')

    err_L2 = None
    if exact_elev_func is not None:
        elev_exact = exact_elev_func(grid, t)
        err2 = (elev_exact - elev)**2 * grid.dx * grid.dy / grid.lx / grid.ly
        err_L2 = numpy.sqrt(numpy.sum(err2))
        print(f'L2 error: {err_L2:5.3e}')

    if runtime_plot:
        plt.ioff()

        if exact_elev_func is not None:
            fig2, ax2 = plt.subplots(nrows=1, ncols=1)
            img2 = ax2.pcolormesh(grid.x_u_1d, grid.y_v_1d, elev_exact.T,
                                  vmin=-vmax, vmax=vmax, cmap='RdBu_r')
            plt.colorbar(img2, label='Elevation')
            ax2.set_title('Exact')

        plt.show()

    return duration, err_L2
