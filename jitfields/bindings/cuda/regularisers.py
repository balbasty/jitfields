from ..common.utils import cinfo, cstrides, bound_as_cname, ctypename
from ..common.bounds import convert_bound
from .utils import (culaunch, get_offset_type, to_cupy, CachedKernel)
import cupy as cp


nocheck = convert_bound['nocheck']


def get_flow_kernel(key):
    func, nbatch, ndim, bound, reduce_t, scalar_t, offset_t, *op = key
    template = func
    template += f'<{nbatch}, {ndim}, '
    if 'relax' not in func:
        template += f"'{op[0]}', "
    template += ctypename(reduce_t) + ', '
    template += ctypename(scalar_t) + ', '
    template += ctypename(offset_t) + ', '
    template += ', '.join(map(bound_as_cname, bound))
    template += '>'
    return template


flow_kernels = CachedKernel('reg_flow.cu', get_flow_kernel)


def get_field_kernel(key):
    func, nbatch, ndim, nc, bound, reduce_t, scalar_t, offset_t, *op = key
    template = func
    template += f'<{nbatch}, {ndim}, {nc}, '
    if 'relax' not in func:
        template += f"'{op[0]}', "
    template += ctypename(reduce_t) + ', '
    template += ctypename(scalar_t) + ', '
    template += ctypename(offset_t) + ', '
    template += ', '.join(map(bound_as_cname, bound))
    template += '>'
    return template


field_kernels = CachedKernel('reg_field.cu', get_field_kernel)


def fixop(op):
    return '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='


def flow_matvec(out, inp, bound, voxel_size,
                absolute, membrane, bending, shears, div,
                op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, ndim) tensor
    inp : (*batch, *spatial, ndim) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : float
    membrane : float
    bending : float
    shears : float
    div : float
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, ndim) tensor
    """
    ndim = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('flow_matvec only implemented up to dimension 3')

    np_inp = to_cupy(inp)
    np_out = to_cupy(out)

    offset_t = get_offset_type(np_inp, np_out)
    reduce_t = cp.float64
    scalar_t = np_inp.dtype

    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, instride = cinfo(np_inp, dtype=offset_t, backend=cp)
    outstride = cstrides(np_out, dtype=offset_t, backend=cp)

    op = fixop(op)
    if bending == div == shears == membrane == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, np_inp, shape, outstride, instride, voxel_size)

    asreduce = (cp.float16 if reduce_t == cp.float16 else
                cp.float32 if reduce_t == cp.float32 else
                cp.float64)
    absolute = asreduce(absolute)
    membrane = asreduce(membrane)
    bending = asreduce(bending)
    shears = asreduce(shears)
    div = asreduce(div)

    func = 'matvec_'
    if bending:
        if div or shears:
            func += 'all'
            args += (absolute, membrane, bending, shears, div)
        else:
            func += 'bending'
            args += (absolute, membrane, bending)
    elif div or shears:
        func += 'lame'
        args += (absolute, membrane, shears, div)
    elif membrane:
        func += 'membrane'
        args += (absolute, membrane)
    elif absolute:
        func += 'absolute'
        args += (absolute,)
    elif op == '=':
        out.zero_()
        return out

    func = flow_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def flow_matvec_rls(out, inp, wgt, bound, voxel_size,
                     absolute, membrane, bending, shears, div,
                     op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, ndim) tensor
    inp : (*batch, *spatial, ndim) tensor
    wgt : (*batch, *spatial) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : float
    membrane : float
    bending : float
    shears : float
    div : float
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, ndim) tensor
    """
    ndim = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('flow_matvec_rls only implemented up to dimension 3')

    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_wgt = to_cupy(wgt)

    offset_t = get_offset_type(np_inp, np_out, np_wgt)
    scalar_t = np_inp.dtype
    reduce_t = cp.float64

    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, instride = cinfo(np_inp, dtype=offset_t, backend=cp)
    outstride = cstrides(np_out, dtype=offset_t, backend=cp)
    wgtstride = cstrides(np_wgt, dtype=offset_t, backend=cp)

    op = fixop(op)
    if bending == div == shears == membrane == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride, voxel_size)

    asreduce = (cp.float16 if reduce_t == cp.float16 else
                cp.float32 if reduce_t == cp.float32 else
                cp.float64)
    absolute = asreduce(absolute)
    membrane = asreduce(membrane)
    bending = asreduce(bending)
    shears = asreduce(shears)
    div = asreduce(div)

    func = 'matvec_'
    if bending:
        if div or shears:
            func += 'all'
            args += (absolute, membrane, bending, shears, div)
        else:
            func += 'bending'
            args += (absolute, membrane, bending)
    elif div or shears:
        func += 'lame'
        args += (absolute, membrane, shears, div)
    elif membrane:
        func += 'membrane'
        args += (absolute, membrane)
    elif absolute:
        func += 'absolute'
        args += (absolute,)
    elif op == '=':
        out.zero_()
        return out

    func += '_jrls'
    func = flow_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def flow_kernel(out, bound, voxel_size,
                absolute, membrane, bending, shears, div,
                op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, ndim, [ndim]) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : float
    membrane : float
    bending : float
    shears : float
    div : float
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, ndim, [ndim]) tensor
    """
    ndim = out.shape[-1]
    nbatch = out.ndim - ndim - 1 - int(shears or div)
    numel = out.shape[:nbatch].numel()
    if ndim > 3:
        raise ValueError('flow_kernel only implemented up to dimension 3')

    np_out = to_cupy(out)

    offset_t = get_offset_type(np_out)
    scalar_t = np_out.dtype
    reduce_t = cp.float64

    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, stride = cinfo(np_out, dtype=offset_t, backend=cp)

    op = fixop(op)
    if bending == div == shears == membrane == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, shape, stride, voxel_size)

    if op == '=':
        out.zero_()

    asreduce = (cp.float16 if reduce_t == cp.float16 else
                cp.float32 if reduce_t == cp.float32 else
                cp.float64)
    absolute = asreduce(absolute)
    membrane = asreduce(membrane)
    bending = asreduce(bending)
    shears = asreduce(shears)
    div = asreduce(div)

    func = 'kernel_'
    if bending:
        if div or shears:
            func += 'all'
            args += (absolute, membrane, bending, shears, div)
        else:
            func += 'bending'
            args += (absolute, membrane, bending)
    elif div or shears:
        func += 'lame'
        args += (absolute, membrane, shears, div)
    elif membrane:
        func += 'membrane'
        args += (absolute, membrane)
    elif absolute:
        func += 'absolute'
        args += (absolute,)
    else:
        return out

    func = flow_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def flow_diag(out, bound, voxel_size,
              absolute, membrane, bending, shears, div,
              op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, ndim) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : float
    membrane : float
    bending : float
    shears : float
    div : float
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, ndim) tensor
    """
    ndim = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('flow_diag only implemented up to dimension 3')

    np_out = to_cupy(out)

    offset_t = get_offset_type(np_out)
    scalar_t = np_out.dtype
    reduce_t = cp.float64

    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, stride = cinfo(np_out, dtype=offset_t, backend=cp)

    op = fixop(op)
    if bending == div == shears == membrane == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, shape, stride, voxel_size)

    asreduce = (cp.float16 if reduce_t == cp.float16 else
                cp.float32 if reduce_t == cp.float32 else
                cp.float64)
    absolute = asreduce(absolute)
    membrane = asreduce(membrane)
    bending = asreduce(bending)
    shears = asreduce(shears)
    div = asreduce(div)

    if op == '=':
        out.zero_()

    func = 'kernel_'
    if bending:
        if div or shears:
            func += 'all'
            args += (absolute, membrane, bending, shears, div)
        else:
            func += 'bending'
            args += (absolute, membrane, bending)
    elif div or shears:
        func += 'lame'
        args += (absolute, membrane, shears, div)
    elif membrane:
        func += 'membrane'
        args += (absolute, membrane)
    elif absolute:
        func += 'absolute'
        args += (absolute,)
    else:
        return out

    func = flow_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def flow_diag_rls(out, wgt, bound, voxel_size,
                  absolute, membrane, bending, shears, div,
                  op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, ndim) tensor
    wgt : (*batch, *spatial) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : float
    membrane : float
    bending : float
    shears : float
    div : float
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, ndim) tensor
    """
    ndim = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('matvec only implemented up to dimension 3')

    np_out = to_cupy(out)
    np_wgt = to_cupy(wgt)

    offset_t = get_offset_type(np_wgt, np_out)
    scalar_t = np_out.dtype
    reduce_t = cp.float64

    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, stride = cinfo(np_out, dtype=offset_t, backend=cp)
    wgtstride = cstrides(np_wgt, dtype=offset_t, backend=cp)

    op = fixop(op)
    if bending == div == shears == membrane == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, np_wgt, shape, stride, wgtstride, voxel_size)

    asreduce = (cp.float16 if reduce_t == cp.float16 else
                cp.float32 if reduce_t == cp.float32 else
                cp.float64)
    absolute = asreduce(absolute)
    membrane = asreduce(membrane)
    bending = asreduce(bending)
    shears = asreduce(shears)
    div = asreduce(div)

    if op == '=':
        out.zero_()

    func = 'kernel_'
    if bending:
        if div or shears:
            func += 'all'
            args += (absolute, membrane, bending, shears, div)
        else:
            func += 'bending'
            args += (absolute, membrane, bending)
    elif div or shears:
        func += 'lame'
        args += (absolute, membrane, shears, div)
    elif membrane:
        func += 'membrane'
        args += (absolute, membrane)
    elif absolute:
        func += 'absolute'
        args += (absolute,)
    else:
        return out

    func += '_jrls'
    func = flow_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def flow_relax_(sol, hes, grd, niter, bound, voxel_size,
                absolute, membrane, bending, shears, div):
    """
    Parameters
    ----------
    sol : (*batch, *spatial, ndim) tensor
    hes : (*batch, *spatial, (ndim*(ndim+1))//2) tensor
    grd : (*batch, *spatial, ndim) tensor
    niter : int
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : float
    membrane : float
    bending : float
    shears : float
    div : float

    Returns
    -------
    sol : (*batch, *spatial, ndim) tensor
    """
    ndim = sol.shape[-1]
    nbatch = sol.ndim - ndim - 1
    numel = sol.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('flow_relax_ only implemented up to dimension 3')

    np_sol = to_cupy(sol)
    np_hes = to_cupy(hes)
    np_grd = to_cupy(grd)

    offset_t = get_offset_type(np_sol, np_hes, np_grd)
    reduce_t = cp.float64
    scalar_t = np_sol.dtype

    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, solstride = cinfo(np_sol, dtype=offset_t, backend=cp)
    hesstride = cstrides(np_hes, dtype=offset_t, backend=cp)
    grdstride = cstrides(np_grd, dtype=offset_t, backend=cp)

    if bending == div == shears == membrane == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, tuple(bound), reduce_t, scalar_t, offset_t)
    args = (np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
            voxel_size)

    asreduce = (cp.float16 if reduce_t == cp.float16 else
                cp.float32 if reduce_t == cp.float32 else
                cp.float64)
    absolute = asreduce(absolute)
    membrane = asreduce(membrane)
    bending = asreduce(bending)
    shears = asreduce(shears)
    div = asreduce(div)

    func = 'relax_'
    if bending:
        if div or shears:
            func += 'all'
            args += (absolute, membrane, bending, shears, div)
        else:
            func += 'bending'
            args += (absolute, membrane, bending)
        subiter = 3**ndim
    elif div or shears:
        func += 'lame'
        args += (absolute, membrane, shears, div)
        subiter = 2**ndim
    elif membrane:
        func += 'membrane'
        args += (absolute, membrane)
        subiter = 2
    elif absolute:
        func += 'absolute'
        args += (absolute,)
        subiter = 1
    else:
        assert False, "no regularization"

    func += '_'
    func = flow_kernels.get(func, *keys)
    for iter in range(niter*subiter):
        culaunch(func, numel, args + (iter,))
    return sol


def flow_relax_rls_(sol, hes, grd, wgt, niter, bound, voxel_size,
                    absolute, membrane, bending, shears, div):
    """
    Parameters
    ----------
    sol : (*batch, *spatial, ndim) tensor
    hes : (*batch, *spatial, (ndim*(ndim+1))//2) tensor
    grd : (*batch, *spatial, ndim) tensor
    wgt : (*batch, *spatial) tensor
    niter : int
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : float
    membrane : float
    bending : float
    shears : float
    div : float

    Returns
    -------
    sol : (*batch, *spatial, ndim) tensor
    """
    ndim = sol.shape[-1]
    nbatch = sol.ndim - ndim - 1
    numel = sol.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('flow_relax_rls_ only implemented up to dimension 3')

    np_sol = to_cupy(sol)
    np_hes = to_cupy(hes)
    np_grd = to_cupy(grd)
    np_wgt = to_cupy(wgt)

    offset_t = get_offset_type(np_sol, np_hes, np_grd, np_wgt)
    reduce_t = cp.float64
    scalar_t = np_sol.dtype

    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, solstride = cinfo(np_sol, dtype=offset_t, backend=cp)
    hesstride = cstrides(np_hes, dtype=offset_t, backend=cp)
    grdstride = cstrides(np_grd, dtype=offset_t, backend=cp)
    wgtstride = cstrides(np_wgt, dtype=offset_t, backend=cp)

    if bending == div == shears == membrane == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, tuple(bound), reduce_t, scalar_t, offset_t)
    args = (np_sol, np_hes, np_grd, np_wgt, shape,
            solstride, hesstride, grdstride, wgtstride, voxel_size)

    asreduce = (cp.float16 if reduce_t == cp.float16 else
                cp.float32 if reduce_t == cp.float32 else
                cp.float64)
    absolute = asreduce(absolute)
    membrane = asreduce(membrane)
    bending = asreduce(bending)
    shears = asreduce(shears)
    div = asreduce(div)

    func = 'relax_'
    if bending:
        if div or shears:
            func += 'all'
            args += (absolute, membrane, bending, shears, div)
        else:
            func += 'bending'
            args += (absolute, membrane, bending)
        subiter = 3**ndim
    elif div or shears:
        func += 'lame'
        args += (absolute, membrane, shears, div)
        subiter = 2**ndim
    elif membrane:
        func += 'membrane'
        args += (absolute, membrane)
        subiter = 2
    elif absolute:
        func += 'absolute'
        args += (absolute,)
        subiter = 1
    else:
        assert False, "no regularization"

    func += '_jrls_'
    func = flow_kernels.get(func, *keys)
    for iter in range(niter*subiter):
        culaunch(func, numel, args + (iter,))
    return sol


def field_matvec(out, inp, bound, voxel_size,
                 absolute, membrane, bending,
                 op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, nc) tensor
    inp : (*batch, *spatial, nc) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, nc) tensor
    """
    ndim = len(voxel_size)
    nc = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('field_matvec only implemented up to dimension 3')

    np_inp = to_cupy(inp)
    np_out = to_cupy(out)

    offset_t = get_offset_type(np_inp, np_out)
    reduce_t = cp.float64
    scalar_t = np_inp.dtype

    np_absolute = cp.ascontiguousarray(cp.asarray(absolute, dtype=reduce_t))
    np_membrane = cp.ascontiguousarray(cp.asarray(membrane, dtype=reduce_t))
    np_bending = cp.ascontiguousarray(cp.asarray(bending, dtype=reduce_t))
    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, instride = cinfo(np_inp, dtype=offset_t, backend=cp)
    outstride = cstrides(np_out, dtype=offset_t, backend=cp)

    op = fixop(op)
    if sum(bending) == sum(membrane) == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, nc, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, np_inp, shape, outstride, instride)
    if any(bending) or any(membrane):
        args = args + (voxel_size,)

    func = 'matvec_'
    if any(bending):
        func += 'bending'
        args += (np_absolute, np_membrane, np_bending)
    elif any(membrane):
        func += 'membrane'
        args += (np_absolute, np_membrane)
    elif any(absolute):
        func += 'absolute'
        args += (np_absolute,)
    elif op == '=':
        out.zero_()
        return out

    func = field_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def field_matvec_rls(out, inp, wgt, bound, voxel_size,
                     absolute, membrane, bending,
                     op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, nc) tensor
    inp : (*batch, *spatial, nc) tensor
    wgt : (*batch, *spatial, 1|nc) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, nc) tensor
    """
    ndim = len(voxel_size)
    nc = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:-1].numel()
    joint = 'j' if wgt.size[-1] > 1 else ''
    if ndim > 3:
        raise ValueError('field_matvec_rls only implemented up to dimension 3')

    np_inp = to_cupy(inp)
    np_out = to_cupy(out)
    np_wgt = to_cupy(wgt)

    offset_t = get_offset_type(np_inp, np_out, np_wgt)
    scalar_t = np_inp.dtype
    reduce_t = cp.float64

    np_absolute = cp.ascontiguousarray(cp.asarray(absolute, dtype=reduce_t))
    np_membrane = cp.ascontiguousarray(cp.asarray(membrane, dtype=reduce_t))
    np_bending = cp.ascontiguousarray(cp.asarray(bending, dtype=reduce_t))
    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, instride = cinfo(np_inp, dtype=offset_t, backend=cp)
    outstride = cstrides(np_out, dtype=offset_t, backend=cp)
    wgtstride = cstrides(np_wgt, dtype=offset_t, backend=cp)

    op = fixop(op)
    if sum(bending) == sum(membrane) == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, nc, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride)
    if any(bending) or any(membrane):
        args = args + (voxel_size,)

    func = 'matvec_'
    if any(bending):
        func += 'bending'
        args += (np_absolute, np_membrane, np_bending)
    elif any(membrane):
        func += 'membrane'
        args += (np_absolute, np_membrane)
    elif any(absolute):
        func += 'absolute'
        args += (np_absolute,)
    elif op == '=':
        out.zero_()
        return out

    func += f'_{joint}rls'
    func = field_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def field_kernel(out, bound, voxel_size,
                 absolute, membrane, bending,
                 op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, nc) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, nc) tensor
    """
    ndim = len(voxel_size)
    nc = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:nbatch].numel()
    if ndim > 3:
        raise ValueError('field_kernel only implemented up to dimension 3')

    np_out = to_cupy(out)

    offset_t = get_offset_type(np_out)
    scalar_t = np_out.dtype
    reduce_t = cp.float64

    np_absolute = cp.ascontiguousarray(cp.asarray(absolute, dtype=reduce_t))
    np_membrane = cp.ascontiguousarray(cp.asarray(membrane, dtype=reduce_t))
    np_bending = cp.ascontiguousarray(cp.asarray(bending, dtype=reduce_t))
    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, stride = cinfo(np_out, dtype=offset_t, backend=cp)

    op = fixop(op)
    if sum(bending) == sum(membrane) == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, nc, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, shape, stride)
    if any(bending) or any(membrane):
        args = args + (voxel_size,)

    if op == '=':
        out.zero_()

    func = 'kernel_'
    if any(bending):
        func += 'bending'
        args += (np_absolute, np_membrane, np_bending)
    elif any(membrane):
        func += 'membrane'
        args += (np_absolute, np_membrane)
    elif any(absolute):
        func += 'absolute'
        args += (np_absolute,)
    else:
        return out

    func = field_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def field_diag(out, bound, voxel_size,
               absolute, membrane, bending,
               op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, nc) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, nc) tensor
    """
    ndim = len(voxel_size)
    nc = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('field_diag only implemented up to dimension 3')

    np_out = to_cupy(out)

    offset_t = get_offset_type(np_out)
    scalar_t = np_out.dtype
    reduce_t = cp.float64

    np_absolute = cp.ascontiguousarray(cp.asarray(absolute, dtype=reduce_t))
    np_membrane = cp.ascontiguousarray(cp.asarray(membrane, dtype=reduce_t))
    np_bending = cp.ascontiguousarray(cp.asarray(bending, dtype=reduce_t))
    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, stride = cinfo(np_out, dtype=offset_t, backend=cp)

    op = fixop(op)
    if sum(bending) == sum(membrane) == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, nc, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, shape, stride)
    if any(bending) or any(membrane):
        args = args + (voxel_size,)

    if op == '=':
        out.zero_()

    func = 'kernel_'
    if any(bending):
        func += 'bending'
        args += (np_absolute, np_membrane, np_bending)
    elif any(membrane):
        func += 'membrane'
        args += (np_absolute, np_membrane)
    elif any(absolute):
        func += 'absolute'
        args += (np_absolute,)
    else:
        return out

    func = field_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def field_diag_rls(out, wgt, bound, voxel_size,
                   absolute, membrane, bending,
                   op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, nc) tensor
    wgt : (*batch, *spatial, 1|nc) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, nc) tensor
    """
    ndim = len(voxel_size)
    nc = out.shape[-1]
    nbatch = out.ndim - ndim - 1
    numel = out.shape[:-1].numel()
    joint = 'j' if wgt.shape[-1] > 1 else ''
    if ndim > 3:
        raise ValueError('field_diag_rls only implemented up to dimension 3')

    np_out = to_cupy(out)
    np_wgt = to_cupy(wgt)

    offset_t = get_offset_type(np_wgt, np_out)
    scalar_t = np_out.dtype
    reduce_t = cp.float64

    np_absolute = cp.ascontiguousarray(cp.asarray(absolute, dtype=reduce_t))
    np_membrane = cp.ascontiguousarray(cp.asarray(membrane, dtype=reduce_t))
    np_bending = cp.ascontiguousarray(cp.asarray(bending, dtype=reduce_t))
    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, stride = cinfo(np_out, dtype=offset_t, backend=cp)
    wgtstride = cstrides(np_wgt, dtype=offset_t, backend=cp)

    op = fixop(op)
    if sum(bending) == sum(membrane) == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, nc, tuple(bound), reduce_t, scalar_t, offset_t, op)
    args = (np_out, np_wgt, shape, stride, wgtstride)
    if any(bending) or any(membrane):
        args = args + (voxel_size,)

    if op == '=':
        out.zero_()

    func = 'kernel_'
    if any(bending):
        func += 'bending'
        args += (np_absolute, np_membrane, np_bending)
    elif any(membrane):
        func += 'membrane'
        args += (np_absolute, np_membrane)
    elif any(absolute):
        func += 'absolute'
        args += (np_absolute,)
    else:
        return out

    func += f'_{joint}rls'
    func = field_kernels.get(func, *keys)
    culaunch(func, numel, args)
    return out


def field_relax_(sol, hes, grd, niter, bound, voxel_size,
                 absolute, membrane, bending):
    """
    Parameters
    ----------
    sol : (*batch, *spatial, nc) tensor
    hes : (*batch, *spatial, (nc*(nc+1))//2) tensor
    grd : (*batch, *spatial, nc) tensor
    niter : int
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]

    Returns
    -------
    sol : (*batch, *spatial, nc) tensor
    """
    ndim = len(voxel_size)
    nc = sol.shape[-1]
    nbatch = sol.ndim - ndim - 1
    numel = sol.shape[:-1].numel()
    if ndim > 3:
        raise ValueError('field_relax_ only implemented up to dimension 3')

    np_sol = to_cupy(sol)
    np_hes = to_cupy(hes)
    np_grd = to_cupy(grd)

    offset_t = get_offset_type(np_sol, np_hes, np_grd)
    reduce_t = cp.float64
    scalar_t = np_sol.dtype

    np_absolute = cp.ascontiguousarray(cp.asarray(absolute, dtype=reduce_t))
    np_membrane = cp.ascontiguousarray(cp.asarray(membrane, dtype=reduce_t))
    np_bending = cp.ascontiguousarray(cp.asarray(bending, dtype=reduce_t))
    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, solstride = cinfo(np_sol, dtype=offset_t, backend=cp)
    hesstride = cstrides(np_hes, dtype=offset_t, backend=cp)
    grdstride = cstrides(np_grd, dtype=offset_t, backend=cp)

    if sum(bending) == sum(membrane) == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, nc, tuple(bound), reduce_t, scalar_t, offset_t)
    args = (np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride)
    if any(bending) or any(membrane):
        args = args + (voxel_size,)

    func = 'relax_'
    if bending:
        func += 'bending'
        args += (np_absolute, np_membrane, np_bending)
        subiter = 3**ndim
    elif membrane:
        func += 'membrane'
        args += (np_absolute, np_membrane)
        subiter = 2
    elif absolute:
        func += 'absolute'
        args += (np_absolute,)
        subiter = 1
    else:
        assert False, "no regularization"

    func += '_'
    func = field_kernels.get(func, *keys)
    for iter in range(niter*subiter):
        culaunch(func, numel, args + (iter,))
    return sol


def field_relax_rls_(sol, hes, grd, wgt, niter, bound, voxel_size,
                     absolute, membrane, bending):
    """
    Parameters
    ----------
    sol : (*batch, *spatial, nc) tensor
    hes : (*batch, *spatial, (nc*(nc+1))//2) tensor
    grd : (*batch, *spatial, nc) tensor
    wgt : (*batch, *spatial, 1|nc) tensor
    niter : int
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]

    Returns
    -------
    sol : (*batch, *spatial, nc) tensor
    """
    ndim = len(voxel_size)
    nc = sol.shape[-1]
    nbatch = sol.ndim - ndim - 1
    numel = sol.shape[:-1].numel()
    joint = 'j' if wgt.shape[-1] > 1 else ''
    if ndim > 3:
        raise ValueError('field_relax_rls_ only implemented up to dimension 3')

    np_sol = to_cupy(sol)
    np_hes = to_cupy(hes)
    np_grd = to_cupy(grd)
    np_wgt = to_cupy(wgt)

    offset_t = get_offset_type(np_sol, np_hes, np_grd, np_wgt)
    reduce_t = cp.float64
    scalar_t = np_sol.dtype

    np_absolute = cp.ascontiguousarray(cp.asarray(absolute, dtype=reduce_t))
    np_membrane = cp.ascontiguousarray(cp.asarray(membrane, dtype=reduce_t))
    np_bending = cp.ascontiguousarray(cp.asarray(bending, dtype=reduce_t))
    voxel_size = cp.ascontiguousarray(cp.asarray(voxel_size, dtype=reduce_t))
    shape, solstride = cinfo(np_sol, dtype=offset_t, backend=cp)
    hesstride = cstrides(np_hes, dtype=offset_t, backend=cp)
    grdstride = cstrides(np_grd, dtype=offset_t, backend=cp)
    wgtstride = cstrides(np_wgt, dtype=offset_t, backend=cp)

    if sum(bending) == sum(membrane) == 0:
        bound = (nocheck,) * ndim
    keys = (nbatch, ndim, nc, tuple(bound), reduce_t, scalar_t, offset_t)
    args = (np_sol, np_hes, np_grd, np_wgt, shape,
            solstride, hesstride, grdstride, wgtstride)
    if any(bending) or any(membrane):
        args = args + (voxel_size,)

    func = 'relax_'
    if any(bending):
        func += 'bending'
        args += (np_absolute, np_membrane, np_bending)
        subiter = 3**ndim
    elif any(membrane):
        func += 'membrane'
        args += (np_absolute, np_membrane)
        subiter = 2
    elif any(absolute):
        func += 'absolute'
        args += (np_absolute,)
        subiter = 1
    else:
        assert False, "no regularization"

    func += f'_{joint}rls_'
    func = field_kernels.get(func, *keys)
    for iter in range(niter*subiter):
        culaunch(func, numel, args + (iter,))
    return sol
