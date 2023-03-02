import torch
from .utils import make_vector
from .sym import sym_solve, sym_solve_


class FieldRegularizer:
    """
    Regularizers for spatial vector fields.

    Such regularizer can be seen as a large matrix L, often with some
    sort of Toeplitz properties (i.e., they can be implemented as a
    convolution).
    """

    def __init__(self):
        self.solver = self.make_solver('fmg')
        self._kernel = None
        self._greens = None

    def set_solver(self, solver, **kwargs):
        """

        Parameters
        ----------
        solver : callable(L, g, [P]) or {'fmg', 'cg'}

        FMG Parameters
        --------------
        max_levels : int, default=16
        nb_cycles : int, default=2
        max_iter : int, default=2
        tolerance : float, default=0
        relax : {'cg', 'gauss', 'jacobi'}, default='cg'

        CG Parameters
        -------------
        max_iter : int, default=16
        tolerance : float, default=1e-5

        """
        self.solver = self.make_solver(solver, **kwargs)
        return self

    def matvec(self, vec, mat=None):
        """Compute the matrix product `L @ x` or `(L + H) @ x`

        This function can be called using either:
            * forward(x)    -> `L @ x`
            * forward(H, x) -> `(L + H) @ x`

        Parameters
        ----------
        vec : (..., *spatial, C)
            Vector field.
        mat : (..., *spatial, K), optional
            Field of positive-definite matrices.
            K is 1 | C | C*(C+1)//2 | C*C

        Returns
        -------
        matvec : (..., *spatial, C)
            `L @ x` or `(L + H) @ x`

        """
        return NotImplemented

    def diagonal(self, nc, **backend):
        """Return the diagonal of the matrix `diag(L)`

        If the matrix is toeplitz-like, only C elements are returned,
        and the full diagonal can be constructed by stacking scaled
        identity matrices.

        Parameters
        ----------
        nc : int
            Number of channels
        dtype : torch.dtype, optional
            Output data type
        device : torch.device, optional
            Output device

        Returns
        -------
        diag : (C,) or (..., *spatial, C) tensor
        """
        return NotImplemented

    def kernel(self, nc, **backend):
        """Return the equivalent convolution kernel.

        Parameters
        ----------
        nc : int
            Number of channels
        dtype : torch.dtype, optional
            Output data type
        device : torch.device, optional
            Output device

        Returns
        -------
        kernel : (*kernel_size, C) or (..., *spatial, *kernel_size, C) tensor
        """
        return NotImplemented

    def save_kernel(self, nc, **backend):
        """Cache the kernel"""
        self._kernel = self.kernel(nc, **backend)

    def greens(self, nc, **backend):
        """Return the Greens function of the convolution kernel (in Fourier domain).

        This function only works if the kernel is Toeplitz-like.

        Parameters
        ----------
        nc : int
            Number of channels
        dtype : torch.dtype, optional
            Output data type
        device : torch.device, optional
            Output device

        Returns
        -------
        greens : (*kernel_size, C) tensor or None
        """
        return NotImplemented

    def save_greens(self, nc, **backend):
        """Cache the Greens kernel"""
        self._greens = self.greens(nc, **backend)

    def solve(self, vec, mat=None):
        """Compute `(H + L).inverse() @ g`

        In some cases, there is no closed-form solution and an iterative
        solver must be used. The default solver may differ across regularizers.

        Parameters
        ----------
        vec : (..., *spatial, C)
            Vector field.
        mat : (..., *spatial, K), optional
            Field of positive-definite matrices.
            K is 1 | C | C*(C+1)//2 | C*C

        Returns
        -------
        sol :  (..., *spatial, C) tensor
            Solution of the linear system.
        """
        return NotImplemented

    def precond(self, vec, mat=None):
        """Apply the preconditioner `(H + diag(L)).inverse() @ x`

        Parameters
        ----------
        vec : (..., *spatial, C)
            Vector field.
        mat : (..., *spatial, K), optional
            Field of positive-definite matrices.
            K is 1 | C | C*(C+1)//2 | C*C

        Returns
        -------
        sol :  (..., *spatial, C) tensor
            Solution of the preconditioning linear system
        """
        return NotImplemented


class FlowRegularizer:
    """
    Regularizers for spatial flow fields.

    Such regularizers can be seen as a large matrix L, often with some
    sort of Toeplitz properties (i.e., they can be implemented as a
    convolution).
    """

    def set_solver(self, solver, **kwargs):
        """

        Parameters
        ----------
        solver : callable(L, g, [P]) or {'fmg', 'cg'}

        FMG Parameters
        --------------
        max_levels : int, default=16
        nb_cycles : int, default=2
        max_iter : int, default=2
        tolerance : float, default=0
        relax : {'cg', 'gauss', 'jacobi'}, default='cg'

        CG Parameters
        -------------
        max_iter : int, default=16
        tolerance : float, default=1e-5

        """
        self.solver = self.make_solver(solver, **kwargs)
        return self

    def matvec(self, vec, mat=None):
        """Compute the matrix product `L @ x` or `(L + H) @ x`

        This function can be called using either:
            * forward(x)    -> `L @ x`
            * forward(H, x) -> `(L + H) @ x`

        Parameters
        ----------
        vec : (..., *spatial, D)
            Flow field.
        mat : (..., *spatial, K), optional
            Field of positive-definite matrices.
            K is 1 | D | D*(D+1)//2 | D*D

        Returns
        -------
        matvec : (..., *spatial, D)
            `L @ x` or `(L + H) @ x`

        """
        return NotImplemented

    def diagonal(self, **backend):
        """Return the diagonal of the matrix `diag(L)`

        If the matrix is toeplitz-like, only C elements are returned,
        and the full diagonal can be constructed by stacking scaled
        identity matrices.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Output data type
        device : torch.device, optional
            Output device

        Returns
        -------
        diag : (D,) or (..., *spatial, D) tensor
        """
        return NotImplemented

    def kernel(self, **backend):
        """Return the equivalent convolution kernel.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Output data type
        device : torch.device, optional
            Output device

        Returns
        -------
        kernel : (*kernel_size, D, D) or (..., *spatial, *kernel_size, D, D) tensor
        """
        return NotImplemented

    def greens(self, **backend):
        """Return the Greens function of the convolution kernel (in Fourier domain).

        This function only works if the kernel is Toeplitz-like.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Output data type
        device : torch.device, optional
            Output device

        Returns
        -------
        greens : (*kernel_size, D, D) tensor or None
        """
        return NotImplemented

    def solve(self, vec, mat=None):
        """Compute `(H + L).inverse() @ g`

        In some cases, there is no closed-form solution and an iterative
        solver must be used. The default solver may differ across regularizers.

        Parameters
        ----------
        vec : (..., *spatial, D)
            Flow field.
        mat : (..., *spatial, K), optional
            Field of positive-definite matrices.
            K is 1 | D | D*(D+1)//2 | D*D

        Returns
        -------
        sol :  (..., *spatial, D) tensor
            Solution of the linear system.
        """
        return NotImplemented

    def precond(self, vec, mat=None):
        """Apply the preconditioner `(H + diag(L)).inverse() @ x`

        Parameters
        ----------
        vec : (..., *spatial, D)
            Flow field.
        mat : (..., *spatial, K), optional
            Field of positive-definite matrices.
            K is 1 | D | D*(D+1)//2 | D*D

        Returns
        -------
        sol :  (..., *spatial, D) tensor
            Solution of the preconditioning linear system
        """
        return NotImplemented


class Absolute(FieldRegularizer):

    def __init__(self, ndim, lam=1, vx=1, weight=None, bound=None):
        """
        Parameters
        ----------
        ndim : int
        lam : [sequence of C] float
        vx : [sequence of ndim] float
        weight : (..., *spatial, C|1) tensor, optional
        bound : unused
        """
        super().__init__()
        self.ndim = ndim
        self.lam = lam
        self.vx = make_vector(vx, ndim).tolist()
        self.weight = weight
        self.bound = None

    def set_bound(self, bound):
        self.bound = None
        return self

    def set_weight(self, weight):
        self.weight = weight
        return self

    def set_vx(self, vx):
        self.vx = make_vector(vx, self.ndim).tolist()
        return self

    def set_lam(self, lam):
        self.lam = lam
        return self

    def matvec(self, x, H=None):
        nc = x.shape[-1]
        lam = make_vector(self.lam, nc, dtype=x.dtype, device=x.device)
        if self.weight is None:
            return absolute_forward(self.ndim, H, x, lam, self.vx)
        else:
            return absolute_forward_weighted(self.ndim, H, x, self.weight, lam, self.vx)

    def diagonal(self, nc, **backend):
        lam = make_vector(self.lam, nc, **backend)
        if self.weight is None:
            return absolute_diag(self.ndim, nc, lam, self.vx)
        else:
            return absolute_diag_weighted(self.ndim, nc, self.weight, lam, self.vx)

    def solve(self, g, H=None):
        nc = g.shape[-1]
        lam = make_vector(self.lam, nc, dtype=g.dtype, device=g.device)
        if self.weight is None:
            return absolute_solve(self.ndim, H, g, lam, self.vx)
        else:
            return absolute_solve_weighted(self.ndim, H, g, self.weight, lam, self.vx)


class SumOfFieldRegularizer(FieldRegularizer):
    """A sum of regularizers"""

    def __init__(self, regularizers):
        self.regularizers = fuse(regularizers)

    def __iter__(self):
        for regularizer in self:
            yield regularizer

    def matvec(self, x, H=None):
        y = 0
        for regularizer in self:
            y += regularizer.forward(x, H)
        return y

    def diagonal(self, nc, **backend):
        d = 0
        for regularizer in self:
            d += regularizer.diagonal(nc, **backend)
        return d

    def kernel(self, nc, **backend):
        k = 0
        for regularizer in self:
            k += regularizer.kernel(nc, **backend)
        return k

    def make_precond(self, nc, H=None, **backend):
        if H is not None:
            backend['dtype'] = H.dtype
            backend['device'] = H.device
        P = self.diagonal(nc, **backend)
        if H is None:
            def precond(g): return g / P
        elif H.shape[-1] in (1, nc):
            P += H
            def precond(g): return g / P
        elif H.shape[-1] == (nc*(nc+1))//2:
            P, D = H.clone(), P
            P[..., :nc] += D
            def precond(g): return sym_solve(P, g)
        elif H.shape[-1] == nc*nc:
            P, D = H.clone(), P
            P = P.reshape([*P.shape[:-1], nc, nc])
            P.diagonal(0, -1, -2).add_(D)
            def precond(g): return torch.linalg.solve(P, g)
        else:
            raise ValueError('Incompatible sizes')
        return precond

    def make_precond_(self, nc, H=None, **backend):
        if H is not None:
            backend['dtype'] = H.dtype
            backend['device'] = H.device
        P = self.diagonal(nc, **backend)
        if H is None:
            def precond(g): return g.div_(P)
        elif H.shape[-1] in (1, nc):
            P += H
            def precond(g): return g.div_(P)
        elif H.shape[-1] == (nc*(nc+1))//2:
            P, D = H.clone(), P
            P[..., :nc] += D
            def precond(g): return sym_solve_(P, g)
        elif H.shape[-1] == nc*nc:
            P, D = H.clone(), P
            P = P.reshape([*P.shape[:-1], nc, nc])
            P.diagonal(0, -1, -2).add_(D)
            def precond(g): return g.copy_(torch.linalg.solve(P, g))
        else:
            raise ValueError('Incompatible sizes')
        return precond

    def precond(self, g, H=None):
        P = self.make_precond(g.shape[-1], H, dtype=g.dtype, device=g.device)
        return P(g)

    def precond_(self, g, H=None):
        P = self.make_precond_(g.shape[-1], H, dtype=g.dtype, device=g.device)
        return P(g)

    def solve(self, g, H=None):
        P = self.make_precond(g.shape[-1], H, dtype=g.dtype, device=g.device)
        A = lambda g: self.matvec(g, H)
        return self.solver(g, A, P)


class SumOfFlowRegularizer(FlowRegularizer):
    """A sum of regularizers"""

    def __init__(self, regularizers):
        self.regularizers = fuse(regularizers)

    def __iter__(self):
        for regularizer in self:
            yield regularizer

    @property
    def ndim(self):
        return self.regularizers[0].ndim

    def matvec(self, x, H=None):
        y = 0
        for regularizer in self:
            y += regularizer.forward(x, H)
        return y

    def diagonal(self, **backend):
        d = 0
        for regularizer in self:
            d += regularizer.diagonal(**backend)
        return d

    def kernel(self, **backend):
        k = 0
        for regularizer in self:
            k += regularizer.kernel(**backend)
        return k

    def make_precond(self, H=None, **backend):
        if H is not None:
            backend['dtype'] = H.dtype
            backend['device'] = H.device
        P = self.diagonal(**backend)
        nd = self.ndim
        if H is None:
            def precond(g): return g / P
        elif H.shape[-1] in (1, nd):
            P += H
            def precond(g): return g / P
        elif H.shape[-1] == (nd*(nd+1))//2:
            P, D = H.clone(), P
            P[..., :nd] += D
            def precond(g): return sym_solve(P, g)
        elif H.shape[-1] == nd*nd:
            P, D = H.clone(), P
            P = P.reshape([*P.shape[:-1], nd, nd])
            P.diagonal(0, -1, -2).add_(D)
            def precond(g): return torch.linalg.solve(P, g)
        else:
            raise ValueError('Incompatible sizes')
        return precond

    def make_precond_(self, H=None, **backend):
        if H is not None:
            backend['dtype'] = H.dtype
            backend['device'] = H.device
        P = self.diagonal(**backend)
        nd = self.ndim
        if H is None:
            def precond(g): return g.div_(P)
        elif H.shape[-1] in (1, nd):
            P += H
            def precond(g): return g.div_(P)
        elif H.shape[-1] == (nd*(nd+1))//2:
            P, D = H.clone(), P
            P[..., :nd] += D
            def precond(g): return sym_solve_(P, g)
        elif H.shape[-1] == nd*nd:
            P, D = H.clone(), P
            P = P.reshape([*P.shape[:-1], nd, nd])
            P.diagonal(0, -1, -2).add_(D)
            def precond(g): return g.copy_(torch.linalg.solve(P, g))
        else:
            raise ValueError('Incompatible sizes')
        return precond

    def precond(self, g, H=None):
        P = self.make_precond(H, dtype=g.dtype, device=g.device)
        return P(g)

    def precond_(self, g, H=None):
        P = self.make_precond_(H, dtype=g.dtype, device=g.device)
        return P(g)

    def solve(self, g, H=None):
        P = self.make_precond(H, dtype=g.dtype, device=g.device)
        A = lambda g: self.matvec(g, H)
        return self.solver(g, A, P)



