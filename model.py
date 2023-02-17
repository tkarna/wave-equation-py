import numpy


class CGrid:
    def __init__(self, nx, ny, xlim=None, ylim=None):
        """
        Regular Arakawa C grid.

        f---v---f
        |       |
        u   t   u
        |       |
        f---v---f

        t point - cell center
        u point - edge center x
        v point - edge center y
        f point - vertices

        """
        if xlim is None:
            xlim = [-1, 1]
        if ylim is None:
            ylim = [-1, 1]
        self.nx = nx
        self.ny = nx
        self.xlim = xlim
        self.ylim = ylim
        self.lx = self.xlim[1] - self.xlim[0]
        self.ly = self.ylim[1] - self.ylim[0]
        self.dx = self.lx/self.nx
        self.dy = self.ly/self.ny

        # coordinates of T points (cell center)
        self.x_t_1d = numpy.linspace(self.xlim[0] + self.dx/2,
                                     self.xlim[1] - self.dx/2, nx)
        self.y_t_1d = numpy.linspace(self.ylim[0] + self.dy/2,
                                     self.ylim[1] - self.dy/2, ny)
        self.x_t_2d, self.y_t_2d = numpy.meshgrid(self.x_t_1d, self.y_t_1d,
                                                  indexing='ij')
        # coordinates of U and V points (edge centers)
        self.x_u_1d = numpy.linspace(self.xlim[0], self.xlim[1], nx + 1)
        self.y_v_1d = numpy.linspace(self.ylim[0], self.ylim[1], ny + 1)

        self.T_shape = (nx, ny)
        self.U_shape = (nx + 1, ny)
        self.V_shape = (nx, ny+1)
        self.F_shape = (self.nx + 1, self.ny + 1)

        self.dofs_T = int(numpy.prod(numpy.asarray(self.T_shape)))
        self.dofs_U = int(numpy.prod(numpy.asarray(self.U_shape)))
        self.dofs_V = int(numpy.prod(numpy.asarray(self.V_shape)))

        print(f'Grid size: {nx} x {ny}')
        print(f'Elevation DOFs: {self.dofs_T}')
        print(f'Velocity  DOFs: {self.dofs_U + self.dofs_V}')
        print(f'Total     DOFs: {self.dofs_T + self.dofs_U + self.dofs_V}')


def run(nx, ny, initial_elev_func, exact_elev_func=None,
        t_end=1.0, t_export=0.02, dt=None, ntimestep=None,
        backend='numpy',
        runtime_plot=False, vmax=0.5):
    """
    Run simulation.
    """
    kwargs = {}
    if backend in ['numpy', 'ramba']:
        import core_numpy as core
        kwargs['backend'] = backend
    elif backend == 'numba':
        import core_numba as core
    elif backend == 'jax':
        import core_jax as core
    else:
        raise ValueError(f'Unknown backend "{backend}"')

    print(f'Using backend: {backend}')
    grid = CGrid(nx, ny)

    out = core.run(
        grid, initial_elev_func, exact_elev_func=exact_elev_func,
        t_end=t_end, t_export=t_export, dt=dt, ntimestep=ntimestep,
        runtime_plot=runtime_plot, vmax=vmax,
        **kwargs
    )
    return out
