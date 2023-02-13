import numpy
import matplotlib.pyplot as plt
import math
import time as time_mod

# options
runtime_plot = True

# constants
g = 9.81
h = 1.0

# run time
t_end = 1.0
t_export = 0.02


def exact_elevation(x, y, t):
    """
    Exact solution for elevation field.

    Returns time-dependent elevation of a 2D standing wave in a rectangular
    domain.
    """
    amp = 0.5
    c = math.sqrt(g * h)
    n = 1
    sol_x = numpy.sin(2 * n * numpy.pi * x / lx)
    m = 1
    sol_y = numpy.sin(2 * m * numpy.pi * y / ly)
    omega = c * numpy.pi * math.sqrt((n/lx)**2 + (m/ly)**2)
    sol_t = numpy.cos(2 * omega * t)
    return amp * sol_x * sol_y * sol_t


def initial_elev(x, y):
    """Set initial condition for water elevation"""
    return exact_elevation(x, y, 0)


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
V_shape = (nx, ny+1)

dofs_T = int(numpy.prod(numpy.asarray(T_shape)))
dofs_U = int(numpy.prod(numpy.asarray(U_shape)))
dofs_V = int(numpy.prod(numpy.asarray(V_shape)))

print(f'Grid size: {nx} x {ny}')
print(f'Elevation DOFs: {dofs_T}')
print(f'Velocity  DOFs: {dofs_U + dofs_V}')
print(f'Total     DOFs: {dofs_T + dofs_U + dofs_V}')

# state variables
elev = numpy.zeros(T_shape, dtype=numpy.float64)
u = numpy.zeros(U_shape, dtype=numpy.float64)
v = numpy.zeros(V_shape, dtype=numpy.float64)

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

# time step
c = math.sqrt(g*h)
alpha = 0.5
dt = alpha * dx / c
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
    dudt[1:-1, :] = -g * (elev[1:, :] - elev[:-1, :])/dx
    dvdt[:, 1:-1] = -g * (elev[:, 1:] - elev[:, :-1])/dy

    # periodic boundary
    dudt[0, :] = - g * (elev[0, :] - elev[-1, :])/dx
    dudt[-1, :] = dudt[0, :]
    dvdt[:, 0] = - g * (elev[:, 0] - elev[:, -1])/dy
    dvdt[:, -1] = dvdt[:, 0]

    # velocity divergence -h div(u)
    delevdt[...] = -h * ((u[1:, :] - u[:-1, :])/dx + (v[:, 1:] - v[:, :-1])/dy)


if runtime_plot:
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    vmax = 0.5
    img = ax.pcolormesh(x_u_1d, y_v_1d, elev.T, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    cb = plt.colorbar(img, label='Elevation')
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

        total_v = float(numpy.sum(elev + h)) * dx * dy
        if initial_v is None:
            initial_v = total_v
        diff_v = total_v - initial_v

        print(f'{i_export:2d} {i:4d} {t:.3f} elev={elev_max:7.5f} u={u_max:7.5f} dV={diff_v: 6.3e}')
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

elev_exact = exact_elevation(x_t_2d, y_t_2d, t)
err_L2 = numpy.sum((elev_exact - elev)**2) * dx * dy / lx / ly
print(f'L2 error: {err_L2:5.3e}')

if runtime_plot:
    plt.ioff()

    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    vmax = 0.5
    img2 = ax2.pcolormesh(x_u_1d, y_v_1d, elev_exact.T, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    cb = plt.colorbar(img2, label='Elevation')
    ax2.set_title('Exact')

    plt.show()
