import numpy.array_api as numpy
import matplotlib.pyplot as plt
import math
import time as time_mod

# options
runtime_plot = False
plot_energy = True
use_vector_invariant_form = True

# constants
g = 9.81
coriolis = 10.0

# run time
t_end = 1.0
t_export = 0.02


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


# grid
nx = 128
ny = 128
xlim = [-1, 1]
ylim = [-1, 1]
lx = xlim[1] - xlim[0]
ly = ylim[1] - ylim[0]
dx = lx/nx
dy = ly/ny

# coordinates of T points (cell center)
x_t_1d = numpy.linspace(xlim[0] + dx/2, xlim[1] - dx/2, nx)
y_t_1d = numpy.linspace(ylim[0] + dy/2, ylim[1] - dy/2, ny)
x_t_2d, y_t_2d = numpy.meshgrid(x_t_1d, y_t_1d, indexing='ij')
# coordinates of U and V points (edge centers)
x_u_1d = numpy.linspace(xlim[0], xlim[1], nx + 1)
y_v_1d = numpy.linspace(ylim[0], ylim[1], ny + 1)

T_shape = (nx, ny)
U_shape = (nx + 1, ny)
V_shape = (nx, ny + 1)
F_shape = (nx + 1, ny + 1)

dofs_T = int(numpy.prod(numpy.asarray(T_shape)))
dofs_U = int(numpy.prod(numpy.asarray(U_shape)))
dofs_V = int(numpy.prod(numpy.asarray(V_shape)))

print(f'Grid size: {nx} x {ny}')
print(f'Elevation DOFs: {dofs_T}')
print(f'Velocity  DOFs: {dofs_U + dofs_V}')
print(f'Total     DOFs: {dofs_T + dofs_U + dofs_V}')

# state variables
dtype = numpy.float64
elev = numpy.zeros(T_shape, dtype=dtype)
u = numpy.zeros(U_shape, dtype=dtype)
v = numpy.zeros(V_shape, dtype=dtype)

# potential vorticity
q = numpy.zeros(F_shape, dtype=dtype)

# energy
ke = numpy.zeros(T_shape, dtype=dtype)
pe = numpy.zeros(T_shape, dtype=dtype)

# bathymetry
h = numpy.zeros(T_shape, dtype=dtype)

# volume fluxes
hu = numpy.zeros_like(u)
hv = numpy.zeros_like(v)

# for advection term
uux = numpy.zeros_like(u)
vuy = numpy.zeros_like(u)
uvx = numpy.zeros_like(v)
vvy = numpy.zeros_like(v)

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
elev[...] = initial_elev(x_t_2d, y_t_2d)
h[...] = bathymetry(x_t_2d, y_t_2d)
pe_offset = 0.5 * g * numpy.mean(h**2)  # pe for elev=0

# time step
c = float(numpy.max(numpy.sqrt(g*h)))
alpha = 0.5
dt = alpha * dx / c
dt = t_export / int(math.ceil(t_export / dt))
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


def rhs(u, v, elev):
    """
    Evaluate right hand side of the equations
    """
    # terms
    # sign convention: positive on rhs

    # pressure gradient -g grad(elev)
    dudt[1:-1, :] = -g * (elev[1:, :] - elev[:-1, :])/dx
    dvdt[:, 1:-1] = -g * (elev[:, 1:] - elev[:, :-1])/dy

    # periodic boundary
    dudt[0, :] = - g * (elev[0, :] - elev[-1, :])/dx
    dudt[-1, :] = dudt[0, :]
    dvdt[:, 0] = - g * (elev[:, 0] - elev[:, -1])/dy
    dvdt[:, -1] = dvdt[:, 0]

    # volume flux divergence -div(H u)
    H = elev + h

    # compute upwind Hu flux at U and V points
    # hu[1:-1, :] = numpy.where(u[1:-1, :] > 0, H[:-1, :], H[1:, :]) * u[1:-1, :]
    # hu[0, :] = numpy.where(u[0, :] > 0, H[-1, :], H[0, :]) * u[0, :]
    # hu[-1, :] = numpy.where(u[-1, :] > 0, H[-1, :], H[0, :]) * u[-1, :]
    # hv[:, 1:-1] = numpy.where(v[:, 1:-1] > 0, H[:, :-1], H[:, 1:]) * v[:, 1:-1]
    # hv[:, 0] = numpy.where(v[:, 0] > 0, H[:, -1], H[:, 0]) * v[:, 0]
    # hv[:, -1] = numpy.where(v[:, -1] > 0, H[:, -1], H[:, 0]) * v[:, -1]

    # Hu flux using mean H
    hu[1:-1, :] = 0.5 * (H[:-1, :] + H[1:, :]) * u[1:-1, :]
    hu[0, :] = 0.5 * (H[-1, :] + H[0, :]) * u[0, :]
    hu[-1, :] = hu[0, :]
    hv[:, 1:-1] = 0.5 * (H[:, :-1] + H[:, 1:]) * v[:, 1:-1]
    hv[:, 0] = 0.5 * (H[:, -1] + H[:, 0]) * v[:, 0]
    hv[:, -1] = hv[:, 0]

    delevdt[...] = -((hu[1:, :] - hu[:-1, :])/dx + (hv[:, 1:] - hv[:, :-1])/dy)

    dudy = numpy.zeros(F_shape, dtype=dtype)  # F point (nx+1, nx+1)
    dudy[:, 1:-1] = (u[:, 1:] - u[:, :-1])/dy
    dudy[:, 0] = (u[:, 0] - u[:, -1])/dy
    dudy[:, -1] = dudy[:, 0]

    dvdx = numpy.zeros(F_shape, dtype=dtype)  # F point (nx+1, nx+1)
    dvdx[1:-1, :] = (v[1:, :] - v[:-1, :])/dx
    dvdx[0, :] = (v[0, :] - v[-1, :])/dx
    dvdx[-1, :] = dvdx[0, :]

    compute_energy(u, v, elev)

    if not use_vector_invariant_form:
        # advection of momentum
        # dudt += U . grad(u) = u dudx + v dudy = uux + vuy
        # dvdt += U . grad(v) = u dvdx + v dvdy = uvx + vvy
        dudx = numpy.zeros((nx + 2, ny))  # T point extended for BC
        dudx[1:-1, :] = (u[1:, :] - u[:-1, :])/dx
        dudx[0, :] = (u[0, :] - u[-1, :])/dx
        dudx[-1, :] = dudx[0, :]
        uux[:, :] = numpy.where(u > 0, dudx[:-1, :], dudx[1:, :]) * u
        dvdy = numpy.zeros((nx, ny + 2))  # T point extended for BC
        dvdy[:, 1:-1] = (v[:, 1:] - v[:, :-1])/dy
        dvdy[:, 0] = (v[:, 0] - v[:, -1])/dy
        dvdy[:, -1] = dvdy[:, 0]
        vvy[:, :] = numpy.where(v > 0, dvdy[:, :-1], dvdy[:, 1:]) * v
        v_at_u = numpy.zeros_like(u)  # U point (nx+1, ny)
        v_av_y = 0.5 * (v[:, 1:] + v[:, :-1])
        v_at_u[1:-1, :] = 0.5 * (v_av_y[1:, :] + v_av_y[:-1, :])
        v_at_u[0, :] = 0.5 * (v_av_y[0, :] + v_av_y[-1, :])
        v_at_u[-1, :] = v_at_u[0, :]
        u_at_v = numpy.zeros_like(v)  # V point (nx, ny+1)
        u_av_x = 0.5 * (u[1:, :] + u[:-1, :])
        u_at_v[:, 1:-1] = 0.5 * (u_av_x[:, 1:] + u_av_x[:, :-1])
        u_at_v[:, 0] = 0.5 * (u_av_x[:, 0] + u_av_x[:, -1])
        u_at_v[:, -1] = u_at_v[:, 0]
        vuy[:, :] = numpy.where(v_at_u > 0, dudy[:, :-1], dudy[:, 1:]) * v_at_u
        uvx[:, :] = numpy.where(u_at_v > 0, dvdx[:-1, :], dvdx[1:, :]) * u_at_v
        dudt[:, :] += -uux - vuy
        dvdt[:, :] += -uvx - vvy

        # Coriolis
        dudt[:, :] += +coriolis*v_at_u
        dvdt[:, :] += -coriolis*u_at_v

    else:
        # total depth at F points
        H_at_f = numpy.zeros(F_shape, dtype=dtype)
        H_at_f[1:-1, 1:-1] = 0.25 * (H[1:, 1:] + H[:-1, 1:] + H[1:, :-1] + H[:-1, :-1])
        H_at_f[0, 1:-1] = 0.25 * (H[0, 1:] + H[-1, 1:] + H[0, :-1] + H[-1, :-1])
        H_at_f[-1, 1:-1] = H_at_f[0, 1:-1]
        H_at_f[1:-1, 0] = 0.25 * (H[1:, 0] + H[:-1, 0] + H[1:, -1] + H[:-1, -1])
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
        qhv = numpy.zeros(U_shape, dtype=dtype)
        qhv[:-1, :] += q_a[:, :] * hv[:, 1:]
        qhv[-1, :] += q_a[0, :] * hv[0, 1:]
        qhv[1:, :] += q_b[:, :] * hv[:, 1:]
        qhv[0, :] += q_b[-1, :] * hv[-1, 1:]
        qhv[1:, :] += q_g[:, :] * hv[:, :-1]
        qhv[0, :] += q_g[-1, :] * hv[-1, :-1]
        qhv[:-1, :] += q_d[:, :] * hv[:, :-1]
        qhv[-1, :] += q_d[0, :] * hv[0, :-1]
        qhu = numpy.zeros(V_shape, dtype=dtype)
        qhu[:, :-1] += q_g[:, :] * hu[1:, :]
        qhu[:, -1] += q_g[:, 0] * hu[1:, 0]
        qhu[:, :-1] += q_d[:, :] * hu[:-1, :]
        qhu[:, -1] += q_d[:, 0] * hu[:-1, 0]
        qhu[:, 1:] += q_a[:, :] * hu[:-1, :]
        qhu[:, 0] += q_a[:, -1] * hu[:-1, -1]
        qhu[:, 1:] += q_b[:, :] * hu[1:, :]
        qhu[:, 0] += q_a[:, -1] * hu[1:, -1]

        dudt[:, :] += +qhv
        dvdt[:, :] += -qhu

        # gradient of ke
        dkedx = numpy.zeros(U_shape, dtype=dtype)
        dkedx[1:-1, :] = (ke[1:, :] - ke[:-1, :])/dx
        dkedx[0, :] = (ke[0, :] - ke[-1, :])/dx
        dkedx[-1, :] = dkedx[0, :]
        dkedy = numpy.zeros(V_shape, dtype=dtype)
        dkedy[:, 1:-1] = (ke[:, 1:] - ke[:, :-1])/dy
        dkedy[:, 0] = (ke[:, 0] - ke[:, -1])/dy
        dkedy[:, -1] = dkedy[:, 0]

        dudt[:, :] += -dkedx
        dvdt[:, :] += -dkedy


if runtime_plot:
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    vmax = 0.15
    img = ax.pcolormesh(x_u_1d, y_v_1d, elev.T, vmin=-vmax, vmax=vmax, cmap=plt.get_cmap('RdBu_r', 61))
    cb = plt.colorbar(img, label='Elevation')
    fig.canvas.draw()
    fig.canvas.flush_events()

t = 0
i_export = 0
next_t_export = 0
compute_energy(u, v, elev)
initial_e = None
ene_series = []
tic = time_mod.perf_counter()
for i in range(nt+1):

    t = i*dt

    if t >= next_t_export:
        elev_max = float(numpy.max(elev))
        u_max = float(numpy.max(u))
        q_max = float(numpy.max(q))

        H = elev + h
        total_ke = float(numpy.sum(H * ke)) * dx * dy
        total_pe = float(numpy.sum(pe)) * dx * dy
        total_e = total_ke + total_pe
        total_v = float(numpy.sum(H)) * dx * dy
        if initial_e is None:
            initial_e = total_e
            initial_v = total_v
        diff_e = total_e - initial_e
        diff_v = total_v - initial_v
        ene_series.append([total_pe, total_ke, total_e])

        print(f'{i_export:2d} {i:4d} {t:.3f} elev={elev_max:7.5f} u={u_max:7.5f} q={q_max:8.5f} dV={diff_v: 6.3e} PE={total_pe:5.3f} KE={total_ke:5.3f} dE={diff_e: 6.3e}')

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
