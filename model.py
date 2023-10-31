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
        self.ny = ny
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


def run(nx, ny, initial_elev_func, backend='numpy', **kwargs):
    """
    Run simulation.
    """
    if backend in ['numpy', 'ramba']:
        import core_numpy as core
        kwargs['backend'] = backend
    elif backend in ['numba', 'numba-opt']:
        import core_numba as core
        kwargs['backend'] = backend
    elif backend in ['jax', 'jax-gpu']:
        device = 'gpu' if backend == 'jax-gpu' else 'cpu'
        import core_jax as core
        kwargs['device'] = device
    else:
        raise ValueError(f'Unknown backend "{backend}"')

    print(f'Using backend: {backend}')
    grid = CGrid(nx, ny)
    datatype = kwargs['datatype']
    print(f'Datatype: {datatype}')

    out = core.run(grid, initial_elev_func, **kwargs)

    l2_error = out[1]
    if l2_error is None:
        return out

    if (nx < 128 or ny < 128):
        print('Skipping correctness test due to small problem size.')
    else:
        tolerance = 1e-2
        if l2_error > tolerance:
            print(f'ERROR: L2 error exceeds tolerance: {l2_error} > {tolerance}')
        else:
            print('SUCCESS')

    return out
