import constant
import math
import matplotlib.pyplot as plt
import time as time_mod
import numpy


def run(grid, initial_elev_func, bathymetry_func,
        exact_elev_func=None,
        t_end=1.0, t_export=0.02, dt=None, ntimestep=None,
        runtime_plot=False, plot_energy=False, vmax=0.5,
        backend='numpy'):
    """
    Run simulation.
    """
    if backend == 'numpy':
        import numpy as npx
    elif backend == 'ramba':
        import ramba as npx
    else:
        raise ValueError(f'Unknown backend "{backend}"')

    # options
    use_vector_invariant_form = True

    # constants
    g = constant.g
    coriolis = 10.0

    # state variables
    dtype = npx.float64
    elev = npx.zeros(grid.T_shape, dtype=dtype)
    u = npx.zeros(grid.U_shape, dtype=dtype)
    v = npx.zeros(grid.V_shape, dtype=dtype)

    # potential vorticity
    q = npx.zeros(grid.F_shape, dtype=dtype)

    # energy
    ke = npx.zeros(grid.T_shape, dtype=dtype)
    pe = npx.zeros(grid.T_shape, dtype=dtype)

    # bathymetry
    h = npx.zeros(grid.T_shape, dtype=dtype)

    # volume fluxes
    hu = npx.zeros_like(u)
    hv = npx.zeros_like(v)

    # state for RK stages
    elev1 = npx.zeros_like(elev)
    elev2 = npx.zeros_like(elev)
    u1 = npx.zeros_like(u)
    u2 = npx.zeros_like(u)
    v1 = npx.zeros_like(v)
    v2 = npx.zeros_like(v)

    # tendecies u += dt*dudt
    dudt = npx.zeros_like(u)
    dvdt = npx.zeros_like(v)
    delevdt = npx.zeros_like(elev)

    # set bathymetry
    h[...] = bathymetry_func(grid.x_t_2d, grid.y_t_2d)
    pe_offset = 0.5 * g * npx.mean(h**2)  # pe for elev=0

    # time step
    if dt is None:
        c = float(npx.max(npx.sqrt(g*h)))
        alpha = 0.5
        dt = alpha * grid.dx / c
        dt = t_export / int(math.ceil(t_export / dt))
    if ntimestep is not None:
        nt = ntimestep
        t_end = nt * dt
    else:
        nt = int(math.ceil(t_end / dt))
    print(f'Time step: {dt} s')
    print(f'Total run time: {t_end} s, {nt} time steps')

    def compute_energy(u, v, elev):
        """
        Compute kinetic and potential energy from model state.
        """
        # kinetic energy, ke = 1/2 |u|^2
        u2 = u**2
        v2 = v**2
        u2_at_t = 0.5 * (u2[1:, :] + u2[:-1, :])
        v2_at_t = 0.5 * (v2[:, 1:] + v2[:, :-1])
        ke[:, :] = 0.5 * (u2_at_t + v2_at_t)
        # potential energy, pe = 1/2 g (elev^2 - h^2) + offset
        pe[:, :] = 0.5 * g * (elev + h) * (elev - h) + pe_offset

    def rhs(u, v, elev, dudt, dvdt, delevdt):
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

        # volume flux divergence -div(H u)
        H = elev + h

        # Hu flux using mean H
        hu[1:-1, :] = 0.5 * (H[:-1, :] + H[1:, :]) * u[1:-1, :]
        hu[0, :] = 0.5 * (H[-1, :] + H[0, :]) * u[0, :]
        hu[-1, :] = hu[0, :]
        hv[:, 1:-1] = 0.5 * (H[:, :-1] + H[:, 1:]) * v[:, 1:-1]
        hv[:, 0] = 0.5 * (H[:, -1] + H[:, 0]) * v[:, 0]
        hv[:, -1] = hv[:, 0]

        delevdt[...] = -((hu[1:, :] - hu[:-1, :])/grid.dx +
                         (hv[:, 1:] - hv[:, :-1])/grid.dy)

        dudy = npx.zeros(grid.F_shape, dtype=dtype)  # F point (nx+1, nx+1)
        dudy[:, 1:-1] = (u[:, 1:] - u[:, :-1])/grid.dy
        dudy[:, 0] = (u[:, 0] - u[:, -1])/grid.dy
        dudy[:, -1] = dudy[:, 0]

        dvdx = npx.zeros(grid.F_shape, dtype=dtype)  # F point (nx+1, nx+1)
        dvdx[1:-1, :] = (v[1:, :] - v[:-1, :])/grid.dx
        dvdx[0, :] = (v[0, :] - v[-1, :])/grid.dx
        dvdx[-1, :] = dvdx[0, :]

        compute_energy(u, v, elev)

        if not use_vector_invariant_form:
            # advection of momentum
            # dudt += U . grad(u) = u dudx + v dudy = uux + vuy
            # dvdt += U . grad(v) = u dvdx + v dvdy = uvx + vvy
            dudx = npx.zeros((grid.nx + 2, grid.ny))  # T point extended for BC
            dudx[1:-1, :] = (u[1:, :] - u[:-1, :])/grid.dx
            dudx[0, :] = (u[0, :] - u[-1, :])/grid.dx
            dudx[-1, :] = dudx[0, :]
            uux = npx.where(u > 0, dudx[:-1, :], dudx[1:, :]) * u
            dvdy = npx.zeros((grid.nx, grid.ny + 2))  # T point extended for BC
            dvdy[:, 1:-1] = (v[:, 1:] - v[:, :-1])/grid.dy
            dvdy[:, 0] = (v[:, 0] - v[:, -1])/grid.dy
            dvdy[:, -1] = dvdy[:, 0]
            vvy = npx.where(v > 0, dvdy[:, :-1], dvdy[:, 1:]) * v
            v_at_u = npx.zeros_like(u)  # U point (nx+1, ny)
            v_av_y = 0.5 * (v[:, 1:] + v[:, :-1])
            v_at_u[1:-1, :] = 0.5 * (v_av_y[1:, :] + v_av_y[:-1, :])
            v_at_u[0, :] = 0.5 * (v_av_y[0, :] + v_av_y[-1, :])
            v_at_u[-1, :] = v_at_u[0, :]
            u_at_v = npx.zeros_like(v)  # V point (nx, ny+1)
            u_av_x = 0.5 * (u[1:, :] + u[:-1, :])
            u_at_v[:, 1:-1] = 0.5 * (u_av_x[:, 1:] + u_av_x[:, :-1])
            u_at_v[:, 0] = 0.5 * (u_av_x[:, 0] + u_av_x[:, -1])
            u_at_v[:, -1] = u_at_v[:, 0]
            vuy = npx.where(v_at_u > 0, dudy[:, :-1], dudy[:, 1:]) * v_at_u
            uvx = npx.where(u_at_v > 0, dvdx[:-1, :], dvdx[1:, :]) * u_at_v
            dudt[:, :] += -uux - vuy
            dvdt[:, :] += -uvx - vvy

            # Coriolis
            dudt[:, :] += +coriolis*v_at_u
            dvdt[:, :] += -coriolis*u_at_v

        else:
            # total depth at F points
            H_at_f = npx.zeros(grid.F_shape, dtype=dtype)
            H_at_f[1:-1, 1:-1] = 0.25 * (
                H[1:, 1:] + H[:-1, 1:] + H[1:, :-1] + H[:-1, :-1]
            )
            H_at_f[0, 1:-1] = 0.25 * (
                H[0, 1:] + H[-1, 1:] + H[0, :-1] + H[-1, :-1]
            )
            H_at_f[-1, 1:-1] = H_at_f[0, 1:-1]
            H_at_f[1:-1, 0] = 0.25 * (
                H[1:, 0] + H[:-1, 0] + H[1:, -1] + H[:-1, -1]
            )
            H_at_f[1:-1, -1] = H_at_f[1:-1, 0]
            H_at_f[0, 0] = 0.25 * (H[0, 0] + H[-1, 0] + H[0, -1] + H[-1, -1])
            H_at_f[0, -1] = H_at_f[-1, 0] = H_at_f[-1, -1] = H_at_f[0, 0]

            # potential vorticity
            q[:, :] = (coriolis - dudy + dvdx) / H_at_f

            # Advection of potential vorticity, Arakawa and Hsu (1990)
            # Define alpha, beta, gamma, delta for each cell in T points
            w = 1./12
            # alpha[i,j+0.5] = 1/12 (q[i,j+1] + q[i,j] + q[i+1,j+1]) ▛
            q_a = w * (q[:-1, 1:] + q[:-1, :-1] + q[1:, 1:])
            # beta[i,j+0.5] = 1/12 (q[i,j+1] + q[i,j] + q[i-1,j+1]) ▜
            q_b = w * (q[1:, 1:] + q[1:, :-1] + q[:-1, 1:])
            # gamma[i,j+0.5] = 1/12 (q[i,j] + q[i,j+1] + q[i-1,j]) ▟
            q_g = w * (q[1:, :-1] + q[1:, 1:] + q[:-1, :-1])
            # delta[i,j+0.5] = 1/12 (q[i,j] + q[i,j+1] + q[i+1,j]) ▙
            q_d = w * (q[:-1, :-1] + q[:-1, 1:] + q[1:, :-1])

            # potential vorticity advection terms
            qhv = npx.zeros(grid.U_shape, dtype=dtype)
            qhv[:-1, :] += q_a[:, :] * hv[:, 1:]
            qhv[-1, :] += q_a[0, :] * hv[0, 1:]
            qhv[1:, :] += q_b[:, :] * hv[:, 1:]
            qhv[0, :] += q_b[-1, :] * hv[-1, 1:]
            qhv[1:, :] += q_g[:, :] * hv[:, :-1]
            qhv[0, :] += q_g[-1, :] * hv[-1, :-1]
            qhv[:-1, :] += q_d[:, :] * hv[:, :-1]
            qhv[-1, :] += q_d[0, :] * hv[0, :-1]
            qhu = npx.zeros(grid.V_shape, dtype=dtype)
            qhu[:, :-1] += q_g[:, :] * hu[1:, :]
            qhu[:, -1] += q_g[:, 0] * hu[1:, 0]
            qhu[:, :-1] += q_d[:, :] * hu[:-1, :]
            qhu[:, -1] += q_d[:, 0] * hu[:-1, 0]
            qhu[:, 1:] += q_a[:, :] * hu[:-1, :]
            qhu[:, 0] += q_a[:, -1] * hu[:-1, -1]
            qhu[:, 1:] += q_b[:, :] * hu[1:, :]
            qhu[:, 0] += q_a[:, -1] * hu[1:, -1]

            dudt[:, :] += qhv
            dvdt[:, :] += -qhu

            # gradient of ke
            dkedx = npx.zeros(grid.U_shape, dtype=dtype)
            dkedx[1:-1, :] = (ke[1:, :] - ke[:-1, :])/grid.dx
            dkedx[0, :] = (ke[0, :] - ke[-1, :])/grid.dx
            dkedx[-1, :] = dkedx[0, :]
            dkedy = npx.zeros(grid.V_shape, dtype=dtype)
            dkedy[:, 1:-1] = (ke[:, 1:] - ke[:, :-1])/grid.dy
            dkedy[:, 0] = (ke[:, 0] - ke[:, -1])/grid.dy
            dkedy[:, -1] = dkedy[:, 0]

            dudt[:, :] += -dkedx
            dvdt[:, :] += -dkedy

    def step(u, v, elev, u1, v1, elev1, u2, v2, elev2, dudt, dvdt, delevdt):
        """
        Execute one SSPRK(3,3) time step
        """
        one_third = 1./3
        two_thirds = 2./3
        rhs(u, v, elev, dudt, dvdt, delevdt)
        u1[...] = u + dt*dudt
        v1[...] = v + dt*dvdt
        elev1[...] = elev + dt*delevdt
        rhs(u1, v1, elev1, dudt, dvdt, delevdt)
        u2[...] = 0.75*u + 0.25*(u1 + dt*dudt)
        v2[...] = 0.75*v + 0.25*(v1 + dt*dvdt)
        elev2[...] = 0.75*elev + 0.25*(elev1 + dt*delevdt)
        rhs(u2, v2, elev2, dudt, dvdt, delevdt)
        u[...] = one_third*u + two_thirds*(u2 + dt*dudt)
        v[...] = one_third*v + two_thirds*(v2 + dt*dvdt)
        elev[...] = one_third*elev + two_thirds*(elev2 + dt*delevdt)

    if backend == 'ramba':
        # warm jit cache
        step(u, v, elev, u1, v1, elev1, u2, v2, elev2, dudt, dvdt, delevdt)

    # initial condition
    elev[...] = npx.asarray(initial_elev_func(grid.x_t_2d, grid.y_t_2d))
    u[...] = 0
    v[...] = 0

    if runtime_plot:
        plt.ion()
        fig, ax_list = plt.subplots(nrows=1, ncols=2,
                                    sharey=True, figsize=(13, 5))
        ax = ax_list[0]
        vmax = 0.15
        img1 = ax.pcolormesh(
            grid.x_u_1d, grid.y_v_1d, numpy.asarray(elev.T),
            vmin=-vmax, vmax=vmax, cmap=plt.get_cmap('RdBu_r', 61)
        )
        plt.colorbar(img1, label='Elevation')
        ax.set_aspect('equal')
        ax = ax_list[1]
        vmax = 0.7
        u_at_t = 0.5 * (u[1:, :] + u[:-1, :])
        img2 = ax.pcolormesh(
            grid.x_u_1d, grid.y_v_1d, numpy.asarray(u_at_t.T),
            vmin=-vmax, vmax=vmax, cmap=plt.get_cmap('RdBu_r', 61)
        )
        plt.colorbar(img2, label='X Velocity')
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    t = 0
    i_export = 0
    next_t_export = 0
    compute_energy(u, v, elev)
    diff_e = None
    diff_v = None
    ene_series = []
    tic = time_mod.perf_counter()
    for i in range(nt+1):

        t = i*dt

        if t >= next_t_export:
            elev_max = float(npx.max(elev))
            u_max = float(npx.max(u))
            q_max = float(npx.max(q))

            H = elev + h
            total_ke = float(npx.sum(H * ke)) * grid.dx * grid.dy
            total_pe = float(npx.sum(pe)) * grid.dx * grid.dy
            total_e = total_ke + total_pe
            total_v = float(npx.sum(H)) * grid.dx * grid.dy
            if diff_e is None:
                initial_e = total_e
                initial_v = total_v
            diff_e = total_e - initial_e
            diff_v = total_v - initial_v
            ene_series.append([total_pe, total_ke, total_e])

            print(
                f'{i_export:2d} {i:4d} {t:.3f} elev={elev_max:7.5f} '
                f'u={u_max:7.5f} q={q_max:8.5f} dV={diff_v: 6.3e} '
                f'PE={total_pe:5.3f} KE={total_ke:5.3f} dE={diff_e: 6.3e}'
            )

            if elev_max > 1e3 or not math.isfinite(elev_max):
                print(f'Invalid elevation value: {elev_max}')
                break
            i_export += 1
            next_t_export = i_export * t_export
            if runtime_plot:
                img1.update({'array': numpy.asarray(elev.T)})
                img2.update(
                    {'array': numpy.asarray(0.5*(u[1:, :] + u[:-1, :])).T}
                )
                fig.canvas.draw()
                fig.canvas.flush_events()

        step(u, v, elev, u1, v1, elev1, u2, v2, elev2, dudt, dvdt, delevdt)

    duration = time_mod.perf_counter() - tic
    print(f'Duration: {duration:.2f} s')

    err_L2 = None
    if exact_elev_func is not None:
        elev_exact = exact_elev_func(grid, t)
        err2 = (elev_exact - elev)**2 * grid.dx * grid.dy / grid.lx / grid.ly
        err_L2 = npx.sqrt(npx.sum(err2))
        print(f'L2 error: {err_L2:7.5e}')

    if runtime_plot:
        plt.ioff()
        plt.show()

    if plot_energy:
        import numpy as np
        ene_series = np.asarray(ene_series)
        pe = ene_series[:, 0]
        ke = ene_series[:, 1]
        te = ene_series[:, 2]
        time = np.arange(pe.shape[0], dtype=dtype) * t_export
        plt.plot(time, te, 'SpringGreen', label='total E')
        plt.plot(time, pe, 'Crimson', label='potential E')
        plt.plot(time, ke, 'RoyalBlue', label='kinetic E')
        plt.axhline(te[0], color='k', zorder=0)
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        plt.show()

    return duration, diff_e, diff_e, err_L2
