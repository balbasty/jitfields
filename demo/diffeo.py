from jitfields import (
    pull, push, grad,
    identity_grid, add_identity_grid_, add_identity_grid,
)
import torch


def matvec(x, y, out=None):
    if out is not None:
        out = out.unsqueeze(-1)
    return x.matmul(y.unsqueeze(-1), out=out).squeeze(-1)


def sym_vecmatvec_(h, x):
    n = x.shape[-1]
    h[..., :n].mul_(x).mul_(x)
    c = n
    for i in range(n):
        for j in range(n+1):
            h[..., c].mul_(x[..., i]).mul_(x[..., j])
            c += 1
    return h


def sym_vecmatvec(h, x, out=None):
    if out is None:
        out = h.clone()
    else:
        out = out.copy_(h)
    return sym_vecmatvec_(out, x)


def sym_matmatmat(h, x, out=None):
    # A = J * H * J'
    # A[i,j] = sum[k,l] J[i,k] H[k,l] J[j,l]
    # A[i,i] = sum[k,l] J[i,k] H[k,l] J[i,l]
    n, m = x.shape[-2:]
    if out is None:
        out = h.new_empty([*h.shape[:-1], n*(n+1)//2])
    for i in range(n):
        out[..., i] = (h[..., :m] * x[..., i, :] * x[..., i, :]).sum(-1)
        c = m
        for k in range(m):
            for l in range(k+1, m):
                out[..., i] += 2 * h[..., c] * x[..., i, k] * x[..., i, l]
                c += 1
        d = n
        for j in range(i+1, n):
            out[..., d] = (h[..., :m] * x[..., i, :] * x[..., j, :]).sum(-1)
            c = m
            for k in range(m):
                for l in range(k+1, m):
                    out[..., d] += h[..., c] * (x[..., i, k] * x[..., j, l] + x[..., j, k] * x[..., i, l])
                    c += 1
            d += 1
    return out


def jacobian(x, bound='dct2', ndim=None, out=None):
    """Compute the Jacobian of a vector field

    Parameters
    ----------
    x : (..., *spatial, nc) tensor
        Input vector field
    bound : bound-like, optional
        Boundary condition
    ndim : int, default=`x.ndim-1`
        Number of spatial dimensions
    out : (..., *spatial, nc, ndim) tensor, optional
        Output placeholder

    Returns
    -------
    jac : (..., *spatial, nc, ndim) tensor
        Jacobian of the vector field

    """
    ndim = ndim or (x.nidm - 1)
    spatial = x.shape[-ndim-1:-1]
    grid = identity_grid(spatial, dtype=x.dtype, device=x.device)
    if out is None:
        out = x.new_empty([*x.shape, ndim])
    for d in range(ndim):
        order = [0]*ndim
        order[d] = 2
        grad(x, grid, order=order, bound=bound, extrapolate=True,
             prefilter=False, out=out[..., d])
    return out


def compose(*flows, bound='dft', order=1, extrapolate=True):
    """Compose a series of displacement fields

    Parameters
    ----------
    u, v, ... : (..., *spatial, ndim) tensor
        Displacement fields, in voxels
    bound : bound-like, optional
        Boundary condition
    order : int, default=1
        Interpolation order
    extrapolate : bool, default=True
        Extrapolate out-of-bound  data

    Returns
    -------
    uov : (..., *spatial, ndim) tensor
        Composed displacement field

    """
    def compose2(u, v):
        uov = add_identity_grid(v)
        uov = pull(u, uov, bound=bound, order=order,
                   extrapolate=extrapolate)
        uov = uov.add_(v)
        return uov
    flows = list(flows)
    flow = flows.pop(-1)
    while flows:
        flow = compose2(flows.pop(-1), flow)
    return flow


def _propagate_gradient(vel, flow, grad, hess=None, **kwargs):
    ndim = vel[0].shape[-1]
    ograd = grad.new_empty([len(vel), *grad.shape])
    if hess is not None:
        ohess = hess.new_empty([len(vel), *hess.shape])

    # Rotate gradient
    jac = jacobian(flow, **kwargs).transpose(-1, -2)
    grad = matvec(jac, grad, out=ograd[-1])
    if hess is not None:
        hess = sym_matmatmat(hess, jac, out=ohess[-1])

    # Propagate backward
    for t in reversed(range(len(vel)-1)):
        jac = jacobian(-vel[t+1], **kwargs).transpose(-1, -2)
        grid = add_identity_grid(vel[t+1])
        grad = push(grad[t+1], grid, **kwargs)
        grad = matvec(jac, grad, out=ograd[t])
        if hess is not None:
            hess = push(hess[t+1], grid, **kwargs)
            hess = sym_matmatmat(hess, jac, out=ohess[t])

    ograd = ograd.movedim(0, -ndim-2)
    if hess is not None:
        ohess = ohess.movedim(0, -ndim-2)
    return ograd, ohess


def propagate_gradhess(vel, grad, hess, flow=None,
                       bound='dft', order=1, extrapolate=True):
    """Propagate gradient and Hessian from integrated flow to velocity flow

    Parameters
    ----------
    vel : (..., nt, *spatial, ndim) tensor
        Series of small displacement fields (velocities)
    grad : (..., *spatial, ndim) tensor
        Gradient with respect to the integrated flow
    hess : (..., *spatial, ndim*(ndim+1)//2) tensor
        Hessian with respect to the integrated flow
    flow : (..., *spatial, ndim) tensor, optional
        Integrated flow
    bound : bound-like, optional
        Boundary condition
    order : int, default=1
        Interpolation order
    extrapolate : bool, default=True
        Extrapolate out-of-bound  data

    Returns
    -------
    grad : (..., nt, *spatial, ndim) tensor
        Gradient wrt series of small displacement fields
    hess : (..., nt, *spatial, ndim*(ndim+1)//2) tensor
        Hessian wrt series of small displacement fields

    """
    ndim = vel.shape[-1]
    kwargs = dict(bound=bound, order=order, extrapolate=extrapolate)
    vel = vel.unbind(-ndim-2)
    if flow is None:
        flow = compose(*vel[:-1], **kwargs)
    else:
        flow = compose(flow, -vel[-1], **kwargs)
    return _propagate_gradient(vel, flow, grad, hess, **kwargs)


def propagate_grad(vel, grad, flow=None,
                   bound='dft', order=1, extrapolate=True):
    """Propagate gradient from integrated flow to velocity flow

    Parameters
    ----------
    vel : (..., nt, *spatial, ndim) tensor
        Series of small displacement fields (velocities)
    grad : (..., *spatial, ndim) tensor
        Gradient with respect to the integrated flow
    bound : bound-like, optional
        Boundary condition
    order : int, default=1
        Interpolation order
    extrapolate : bool, default=True
        Extrapolate out-of-bound  data

    Returns
    -------
    grad : (..., nt, *spatial, ndim) tensor
        Gradient wrt series of small displacement fields

    """
    ndim = vel.shape[-1]
    kwargs = dict(bound=bound, order=order, extrapolate=extrapolate)
    vel = vel.unbind(-ndim-2)
    if flow is None:
        flow = compose(*vel[:-1], **kwargs)
    else:
        flow = compose(flow, -vel[-1], **kwargs)
    return _propagate_gradient(vel, flow, grad, **kwargs)



def regress(target, mask, lam=10, prm=None, fov=None, nt=8):
    fov = dict(fov or {})
    fov.setdefault('bound', 'dft')
    fov.setdefault('order', 1)
    fov.setdefault('extrapolate', True)
    prm = dict(prm or {})
    prm.setdefault('absolute', 1e-3)
    prm.setdefault('membrane', 1)

    vel = target.new_zeros([nt, *target.shape])
    flow = torch.zeros_like(target)

    for n_iter in range(32):





