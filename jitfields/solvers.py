__all__ = ['ConjugateGradient', 'PrepCG', 'MultiGrid']

import torch
from .resize import Prolong, Restrict


def _dot(x, y, out=None):
    return torch.dot(x.flatten(), y.flatten(), out=out)


class ConjugateGradient:
    """
    Conjugate-gradient solver for linear systems of the form `A @ x = b`
    """

    def __init__(self, max_iter=32, tol=1e-4, dot=None):
        """

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for early stopping
        dot : callable(tensor, tensor) -> tensor
            Function to use for dot product
        """
        self.max_iter = max_iter
        self.tol = tol
        self.dot = dot

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, target, forward, precond=None, init=None, inplace=False):
        """Solve for `x` in `A @ x = b`

        Parameters
        ----------
        target : tensor
            Target vector `b`
        forward : callable(tensor) -> tensor
            Forward matrix-vector product `A`
        precond : callable(tensor) -> tensor, optional
            Preconditioning matrix-vector product `P`
        init : tensor, optional
            Initial value for the solution `x`

        Returns
        -------
        solution : tensor
            Solution `x`

        """
        x = (torch.zeros_like(target) if init is None
             else init if inplace else init.clone())
        g = target
        A = forward
        P = precond or torch.clone
        dot = self.dot or _dot

        # init
        r = A(x)                    # r  = A @ x
        torch.sub(g, r, out=r)      # r  = g - r
        z = P(r)                    # z  = P @ r
        rz = dot(r, z)              # rz = r' @ z
        p = z.clone()               # Initial conjugate directions p

        for n_iter in range(self.max_iter):
            Ap = forward(p)                 # Ap = A @ p
            pAp = dot(p, Ap)                # p' @ A @ p
            a = rz / pAp.clamp_min_(1e-12)  # α = (r' @ z) / (p' @ A @ p)

            if self.tol:
                obj = a * (a * pAp + 2 * dot(p, r))
                obj *= a.numel() / r.numel()
                obj = obj.mean()
                if obj < self.tol:
                    break

            x.addcmul_(a, p)                # x += α * p
            r.addcmul_(a, Ap, value=-1)     # r -= α * (A @ p)
            z = P(r)                        # z  = P @ r
            rz0 = rz
            rz = dot(r, z)                  # rz = r' @ z
            b = rz / rz0                    # β
            torch.addcmul(z, b, p, out=p)   # p = z + β * p

        return x

    def solve_(self, init, target, forward, precond=None):
        """Solve for `x` in `A @ x = b` in-place.

        Parameters
        ----------
        init : tensor
            Initial value for the solution `x`
        target : tensor
            Target vector `b`
        forward : callable(tensor) -> tensor
            Forward matrix-vector product `A`
        precond : callable(tensor) -> tensor, optional
            Preconditioning matrix-vector product `P`

        Returns
        -------
        solution : tensor
            Solution `x`

        """
        return self.solve(target, forward, precond, init, inplace=True)

    def advanced_solve_(self, init, target, forward, precond=None,
                        p=None, r=None, z=None):
        """Solve for `x` in `A @ x = b`

        Parameters
        ----------
        init : tensor
            Initial value for the solution `x`
        target : tensor
            Target vector `b`
        forward : callable(tensor, out=tensor) -> tensor
            Forward matrix-vector product `A`
        precond : callable(tensor, out=tensor) -> tensor, optional
            Preconditioning matrix-vector product `P`

        Returns
        -------
        solution : tensor
            Solution `x`

        """
        x = init
        g = target
        A = forward
        P = precond or (lambda z, out: z)
        p = torch.empty_like(x) if p is None else p
        r = torch.empty_like(x) if r is None else r
        z = torch.empty_like(x) if z is None else z
        Ap = z  # can use same buffer as z
        dot = self.dot or _dot

        # init
        r = A(x, out=r)                 # r  = A @ x
        r = torch.sub(g, r, out=r)      # r  = g - r
        z = P(r, out=z)                 # z  = P @ r
        rz = dot(r, z)                  # rz = r' @ z
        p = p.copy_(z)                  # Initial conjugate directions p

        for n_iter in range(self.max_iter):
            Ap = forward(p, out=Ap)         # Ap = A @ p    (can use same buffer as z)
            pAp = dot(p, Ap)                # p' @ A @ p
            a = rz / pAp.clamp_min(1e-12)   # α = (r' @ z) / (p' @ A @ p)

            if True:  # self.tol:
                obj = a * (a * pAp + 2 * dot(p, r))
                obj *= a.numel() / r.numel()
                obj = obj.mean()
                print(obj)
                # if obj < self.tol:
                #     break

            x.addcmul_(a, p)                # x += α * p

            if n_iter == self.max_iter-1:
                break

            r.addcmul_(a, Ap, value=-1)     # r -= α * (A @ p)
            z = P(r, out=z)                 # z  = P @ r
            rz0, rz = rz, dot(r, z)         # rz = r' @ z
            b = rz / rz0                    # β
            torch.addcmul(z, b, p, out=p)   # p = z + β * p

        return x


class PrepCG:
    """
    Conjugate-Gradient solver that stores `A` and `b` at construction
    """

    def __init__(self, target, forward, precond=None,
                 max_iter=32, tol=1e-4, dot=None,
                 p=None, r=None, z=None):
        """

        Parameters
        ----------
        target : tensor
            Target vector `b`
        forward : callable(tensor, out=tensor) -> tensor
            Forward matrix-vector product `A`
        precond : callable(tensor, out=tensor) -> tensor, optional
            Preconditioning matrix-vector product `P`
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for early stopping
        dot : callable(tensor, tensor) -> tensor
            Function to use for dot product
        """
        self.target = target
        self.forward = forward
        self.precond = precond
        self.p = torch.empty_like(target) if p is None else p
        self.r = torch.empty_like(target) if r is None else r
        self.z = torch.empty_like(target) if z is None else z
        self._solver = ConjugateGradient(max_iter, tol, dot)

    @property
    def max_iter(self):
        return self._solver.max_iter

    @max_iter.setter
    def max_iter(self, value):
        self._solver.max_iter = value

    @property
    def tol(self):
        return self._solver.tol

    @tol.setter
    def tol(self, value):
        self._solver.tol = value

    @property
    def dot(self):
        return self._solver.dot

    @dot.setter
    def dot(self, value):
        self._solver.dot = value

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, init=None, inplace=False):
        return self._solver.solve(self.target, self.forward, self.precond,
                                  init, inplace)

    def solve_(self, init):
        return self._solver.solve_(init, self.target, self.forward, self.precond)

    def advanced_solve_(self, init):
        return self._solver.advanced_solve_(
            init, self.target, self.forward,  self.precond,
            self.p, self.r, self.z)


class MultiGridCG:
    """
    Multi-grid solver for linear systems of the form `(H+L)*x = b`, where
    `L` has a multi-resolution interpretation (e.g., spatial regularizer)
    """

    def __init__(self, nb_cycles=2, max_iter=2, tol=0,
                 prolong=None, restrict=None):
        """

        Parameters
        ----------
        nb_cycles : int
            Number of W cycles
        """
        self.nb_cycles = nb_cycles
        self.solver = ConjugateGradient(max_iter=max_iter, tol=tol)
        self.prolong = prolong
        self.restrict = restrict

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, target, forward, precond=None, init=None, inplace=False):
        """

        Parameters
        ----------
        target : list[tensor]
        forward : list[callable(tensor) -> tensor]
        precond : list[callable(tensor) -> tensor], optional
        init : tensor, optional
        inplace : bool, default=False

        Returns
        -------
        solution : tensor

        """
        solve_ = self.solver.solve_
        prolong = self.prolong or Prolong(target[0].ndim)
        restrict = self.restrict or Restrict(target[0].ndim)

        def residuals(A, x, g, r):
            return torch.sub(A(x), g, out=r)

        g = target
        A = forward
        P = precond or ([torch.clone] * len(g))
        x = (g[0].clone() if init is None else init.clone() if not inplace else init)
        N = len(target)

        x = [x] + list(map(torch.empty_like, g[1:]))
        for n in range(N-1):
            restrict(x[n], x[n+1])
        r = list(map(torch.empty_like, g))

        # initial solve at coarsest resolution
        solve_(x[-1], g[-1], A[-1], P[-1])

        for n_base in reversed(range(N-1)):
            prolong(x[n_base+1], x[n_base])

            for n_cycle in range(self.nb_cycles):
                for n in range(n_base, N-1):
                    solve_(x[n], g[n], A[n], P[n])
                    residuals(A[n], x[n], g[n], r[n])
                    restrict(r[n], g[n+1])
                    x[n+1].zero_()

                solve_(x[-1], g[-1], A[-1], P[-1])

                for n in reversed(range(n_base, N-1)):
                    prolong(x[n+1], r[n])
                    x[n] += r[n]
                    solve_(x[n], g[n], A[n], P[n])

        return x[0]

    def solve_(self, init, target, forward, precond=None):
        return self.solve(target, forward, precond, init, inplace=True)

    def advanced_solve_(self, init, target, forward, precond=None):
        """

        Parameters
        ----------
        init : tensor
        target : list[tensor]
        forward : list[callable(tensor) -> tensor]
        precond : list[callable(tensor) -> tensor], optional

        Returns
        -------
        solution : tensor

        """
        prolong = self.prolong or Prolong(target[0].ndim)
        restrict = self.restrict or Restrict(target[0].ndim)

        def residuals(A, x, g, r):
            return torch.sub(A(x), g, out=r)

        g = target
        A = forward
        P = precond or ([lambda x, out: out.copy_(x)] * len(g))
        x = init
        N = len(target)

        mx = self.solver.max_iter
        tol = self.solver.tol
        solve_ = [PrepCG(g1, A1, P1, mx, tol).advanced_solve_
                  for g1, A1, P1 in zip(g, A, P)]

        x = [x] + list(map(torch.empty_like, g[1:]))
        for n in range(N-1):
            restrict(x[n], x[n+1])
        r = list(map(torch.empty_like, g))

        # initial solve at coarsest resolution
        solve_[-1](x[-1])

        for n_base in reversed(range(N-1)):
            prolong(x[n_base+1], x[n_base])

            for n_cycle in range(self.nb_cycles):
                for n in range(n_base, N-1):
                    solve_[n](x[n])
                    residuals(A[n], x[n], g[n], r[n])
                    restrict(r[n], g[n+1])
                    x[n+1].zero_()

                solve_[-1](x[-1])

                for n in reversed(range(n_base, N-1)):
                    prolong(x[n+1], r[n])
                    x[n] += r[n]
                    solve_[n](x[n])

        return x[0]


class MultiGrid:
    """
    Multi-grid solver for linear systems of the form `(H+L)*x = b`, where
    `L` has a multi-resolution interpretation (e.g., spatial regularizer)
    """

    def __init__(self, nb_cycles=2, prolong=None, restrict=None):
        """

        Parameters
        ----------
        nb_cycles : int
            Number of W cycles
        """
        self.nb_cycles = nb_cycles
        self.prolong = prolong
        self.restrict = restrict

    def solve_(self, init, target, forward, solve_):
        """

        Parameters
        ----------
        init : tensor
        target : list[tensor]
        forward : list[callable(tensor) -> tensor]
        solve_ : list[callable(tensor) -> tensor]

        Returns
        -------
        solution : tensor

        """
        prolong = self.prolong or Prolong(target[0].ndim)
        restrict = self.restrict or Restrict(target[0].ndim)

        def residuals(A, x, g, r):
            return torch.sub(A(x), g, out=r)

        g = target
        A = forward
        x = init
        N = len(target)

        x = [x] + list(map(torch.empty_like, g[1:]))
        for n in range(N-1):
            restrict(x[n], x[n+1])
        r = list(map(torch.empty_like, g))

        # initial solve at coarsest resolution
        solve_[-1](x[-1])

        for n_base in reversed(range(N-1)):
            prolong(x[n_base+1], x[n_base])
            foo = 0

            for n_cycle in range(self.nb_cycles):
                for n in range(n_base, N-1):
                    solve_[n](x[n])
                    residuals(A[n], x[n], g[n], r[n])
                    restrict(r[n], g[n+1])
                    x[n+1].zero_()

                solve_[-1](x[-1])

                for n in reversed(range(n_base, N-1)):
                    prolong(x[n+1], r[n])
                    x[n] += r[n]
                    solve_[n](x[n])

        return x[0]
