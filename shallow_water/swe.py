import numpy.array_api as numpy
import matplotlib.pyplot as plt
import math

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
nx = 256
ny = 256
xlim = [-1, 1]
ylim = [-1, 1]
lx = xlim[1] - xlim[0]
ly = ylim[1] - ylim[0]
dx = lx/nx
dy = ly/ny

# coordinates of T points (cell center)
x_t_1d = numpy.linspace(xlim[0] + dx/2, xlim[1] - dx/2, nx)
y_t_1d = numpy.linspace(ylim[0] + dy/2, ylim[1] - dy/2, ny)
x_t_2d, y_t_2d = numpy.meshgrid(x_t_1d, y_t_1d, indexing='xy')
# coordinates of U and V points (edge centers)
x_u_1d = numpy.linspace(xlim[0], xlim[1], nx + 1)
y_v_1d = numpy.linspace(ylim[0], ylim[1], ny + 1)

T_shape = (nx, ny)
U_shape = (nx + 1, ny)
V_shape = (nx, ny+1)

# state variables
elev = numpy.zeros(T_shape, dtype=numpy.float64)
u = numpy.zeros(U_shape, dtype=numpy.float64)
v = numpy.zeros(V_shape, dtype=numpy.float64)

# bathymetry
h = numpy.zeros(T_shape, dtype=numpy.float64)

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

# time step
c = float(numpy.max(numpy.sqrt(g*h)))
alpha = 0.5
dt = alpha * dx / c
dt = t_export / int(math.ceil(t_export / dt))
nt = int(math.ceil(t_end / dt))
print(f'Time step: {dt} s')


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
    dudt[0, :] = - g * (elev[1, :] - elev[-1, :])/dx
    dudt[-1, :] = dudt[0, :]
    dvdt[:, 0] = - g * (elev[:, 1] - elev[:, -1])/dy
    dvdt[:, -1] = dvdt[:, 0]

    # volume flux divergence -div(H u)
    H = elev + h
    # compute upwind Hu flux at U and V points
    hu[1:-1, :] = numpy.where(u[1:-1, :] > 0, H[:-1, :], H[1:, :]) * u[1:-1, :]
    hu[0, :] = numpy.where(u[0, :] > 0, H[-1, :], H[0, :]) * u[0, :]
    hu[-1, :] = numpy.where(u[-1, :] > 0, H[-1, :], H[0, :]) * u[-1, :]
    hv[:, 1:-1] = numpy.where(v[:, 1:-1] > 0, H[:, :-1], H[:, 1:]) * v[:, 1:-1]
    hv[:, 0] = numpy.where(v[:, 0] > 0, H[:, -1], H[:, 0]) * v[:, 0]
    hv[:, -1] = numpy.where(v[:, -1] > 0, H[:, -1], H[:, 0]) * v[:, -1]
    delevdt[...] = -((hu[1:, :] - hu[:-1, :])/dx + (hv[:, 1:] - hv[:, :-1])/dy)

    # advection of momentum
    # dudt += U . grad(u) = u dudx + v dudy = uux + vuy
    # dvdt += U . grad(v) = u dvdx + v dvdy = uvx + vvy
    dudx = (u[1:, :] - u[:-1, :])/dx  # T point (nx, ny)
    uux[1:-1, :] = numpy.where(u[1:-1, :] > 0, dudx[:-1, :], dudx[1:, :]) * u[1:-1, :]
    uux[0, :] = numpy.where(u[0, :] > 0, dudx[-1, :], dudx[0, :]) * u[0, :]
    uux[-1, :] = uux[0, :]  # periodic => u[0, :] == u[-1, :]
    dvdy = (v[:, 1:] - v[:, :-1])/dy  # T point (nx, ny)
    vvy[:, 1:-1] = numpy.where(v[:, 1:-1] > 0, dvdy[:, :-1], dvdy[:, 1:]) * v[:, 1:-1]
    vvy[:, 0] = numpy.where(v[:, 0] > 0, dvdy[:, -1], dvdy[:, 0]) * v[:, 0]
    vvy[:, -1] = vvy[:, 0]  # periodic
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
    dudy = (u[:, 1:] - u[:, :-1])/dy  # F point  (nx+1, ny-1)
    vuy[:, 1:-1] = numpy.where(v_at_u[:, 1:-1] > 0, dudy[:, :-1], dudy[:, 1:]) * v_at_u[:, 1:-1]
    vuy[:, 0] = numpy.where(v_at_u[:, 0] > 0, (u[:, 0] - u[:, -1])/dy, dudy[:, 0]) * v_at_u[:, 0]
    vuy[:, -1] = numpy.where(v_at_u[:, -1] > 0, dudy[:, -1], (u[:, 0] - u[:, -1])/dy) * v_at_u[:, -1]
    dvdx = (v[1:, :] - v[:-1, :])/dx  # F point  (nx-1, ny+1)
    uvx[1:-1, :] = numpy.where(u_at_v[1:-1, :] > 0, dvdx[:-1, :], dvdx[1:, :]) * u_at_v[1:-1, :]
    uvx[0, :] = numpy.where(u_at_v[0, :] > 0, (v[0, :] - v[-1, :])/dx, dvdx[0, :]) * u_at_v[0, :]
    uvx[-1, :] = numpy.where(u_at_v[-1, :] > 0, dvdx[-1, :], (v[0, :] - v[-1, :])/dx) * u_at_v[-1, :]
    dudt[:, :] += -uux - vuy
    dvdt[:, :] += -uvx - vvy

    # Coriolis
    dudt[:, :] += -coriolis*v_at_u
    dvdt[:, :] += coriolis*u_at_v


plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=1)
vmax = 0.2
img = ax.pcolormesh(x_u_1d, y_v_1d, elev, vmin=-vmax, vmax=vmax, cmap=plt.get_cmap('RdBu_r', 61))
cb = plt.colorbar(img, label='Elevation')
fig.canvas.draw()
fig.canvas.flush_events()

t = 0
i_export = 0
next_t_export = 0
for i in range(nt+1):

    t = i*dt

    if t >= next_t_export:
        elev_max = float(numpy.max(elev))
        print(f'{i:04d} {t:.3f} elev={elev_max:9.5f}')
        if elev_max > 1e3:
            print('Invalid elevation value')
            break
        i_export += 1
        next_t_export = i_export * t_export
        img.update({'array': elev})
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


plt.ioff()
plt.show()
