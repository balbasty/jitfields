from .utils import cwrap
from ..common.utils import cinfo, ctypename, bound_template
import cppyy
import numpy as np
from .utils import include

include()
cppyy.include('reg_flow.hpp')
cppyy.include('reg_field.hpp')


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
    if ndim > 3:
        raise ValueError('flow_matvec only implemented up to dimension 3')

    np_inp = inp.numpy()
    np_out = out.numpy()

    offset_t = np.int64
    reduce_t = np.float64
    scalar_t = np_inp.dtype

    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)
    shape, instride = cinfo(np_inp, dtype=offset_t)
    _, outstride = cinfo(np_out, dtype=offset_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if bending == div == shears == membrane == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if bending:
        if div or shears:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_all')[template])
            func(np_out, np_inp, shape, outstride, instride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_bending')[template])
            func(np_out, np_inp, shape, outstride, instride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_lame')[template])
        func(np_out, np_inp, shape, outstride, instride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_membrane')[template])
        func(np_out, np_inp, shape, outstride, instride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_absolute')[template])
        func(np_out, np_inp, shape, outstride, instride,
             voxel_size, absolute)
    elif op == '=':
        out.zero_()

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
    if ndim > 3:
        raise ValueError('flow_matvec_rls only implemented up to dimension 3')

    np_inp = inp.numpy()
    np_out = out.numpy()
    np_wgt = wgt.numpy()

    offset_t = np.int64
    scalar_t = np_inp.dtype
    reduce_t = np.float64

    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)
    shape, instride = cinfo(np_inp, dtype=offset_t)
    _, outstride = cinfo(np_out, dtype=offset_t)
    _, wgtstride = cinfo(np_wgt, dtype=offset_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if bending == div == shears == membrane == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if bending:
        if div or shears:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_all_jrls')[template])
            func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_bending_jrls')[template])
            func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_lame_jrls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_membrane_jrls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'matvec_absolute_jrls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             voxel_size, absolute)
    elif op == '=':
        out.zero_()

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
    if ndim > 3:
        raise ValueError('flow_kernel only implemented up to dimension 3')

    np_out = out.numpy()

    offset_t = np.int64
    scalar_t = np_out.dtype
    reduce_t = np.float64

    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)
    shape, stride = cinfo(np_out, dtype=offset_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if bending == div == shears == membrane == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if op == '=':
        out.zero_()
    if bending:
        if div or shears:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'kernel_all')[template])
            func(np_out, shape, stride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'kernel_bending')[template])
            func(np_out, shape, stride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'kernel_lame')[template])
        func(np_out, shape, stride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'kernel_membrane')[template])
        func(np_out, shape, stride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'kernel_absolute')[template])
        func(np_out, shape, stride,
             voxel_size, absolute)

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
    if ndim > 3:
        raise ValueError('flow_diag only implemented up to dimension 3')

    np_out = out.numpy()

    offset_t = np.int64
    scalar_t = np_out.dtype
    reduce_t = np.float64

    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)
    shape, stride = cinfo(np_out, dtype=offset_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if bending == div == shears == membrane == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if op == '=':
        out.zero_()
    if bending:
        if div or shears:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_all')[template])
            func(np_out, shape, stride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_bending')[template])
            func(np_out, shape, stride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_lame')[template])
        func(np_out, shape, stride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_membrane')[template])
        func(np_out, shape, stride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_absolute')[template])
        func(np_out, shape, stride,
             voxel_size, absolute)

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
    if ndim > 3:
        raise ValueError('matvec only implemented up to dimension 3')

    np_out = out.numpy()
    np_wgt = wgt.numpy()

    offset_t = np.int64
    scalar_t = np_out.dtype
    reduce_t = np.float64

    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)
    shape, stride = cinfo(np_out, dtype=offset_t)
    _, wgtstride = cinfo(np_wgt, dtype=offset_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if bending == div == shears == membrane == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if op == '=':
        out.zero_()
    if bending:
        if div or shears:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_all')[template])
            func(np_out, np_wgt, shape, stride, wgtstride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_bending')[template])
            func(np_out, np_wgt, shape, stride, wgtstride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_lame')[template])
        func(np_out, np_wgt, shape, stride, wgtstride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_membrane')[template])
        func(np_out, np_wgt, shape, stride, wgtstride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'diag_absolute')[template])
        func(np_out, np_wgt, shape, stride, wgtstride,
             voxel_size, absolute)

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
    if ndim > 3:
        raise ValueError('flow_relax_ only implemented up to dimension 3')

    np_sol = sol.numpy()
    np_hes = hes.numpy()
    np_grd = grd.numpy()

    offset_t = np.int64
    reduce_t = np.float64
    scalar_t = np_sol.dtype

    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)
    shape, solstride = cinfo(np_sol, dtype=offset_t)
    _, hesstride = cinfo(np_hes, dtype=offset_t)
    _, grdstride = cinfo(np_grd, dtype=offset_t)

    template = f"{nbatch}, {ndim}"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if bending == div == shears == membrane == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if bending:
        if div or shears:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_all_')[template])
            func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
                 voxel_size, absolute, membrane, bending, shears, div, niter)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_bending_')[template])
            func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
                 voxel_size, absolute, membrane, bending, niter)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_lame_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, membrane, shears, div, niter)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_membrane_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, membrane, niter)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_absolute_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, niter)

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
    if ndim > 3:
        raise ValueError('flow_relax_rls_ only implemented up to dimension 3')

    np_sol = sol.numpy()
    np_hes = hes.numpy()
    np_grd = grd.numpy()
    np_wgt = wgt.numpy()

    offset_t = np.int64
    reduce_t = np.float64
    scalar_t = np_sol.dtype

    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)
    shape, solstride = cinfo(np_sol, dtype=offset_t)
    _, hesstride = cinfo(np_hes, dtype=offset_t)
    _, grdstride = cinfo(np_grd, dtype=offset_t)
    _, wgtstride = cinfo(np_wgt, dtype=offset_t)

    template = f"{nbatch}, {ndim}"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if bending == div == shears == membrane == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if bending:
        if div or shears:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_all_jrls_')[template])
            func(np_sol, np_hes, np_grd, np_wgt,
                 shape, solstride, hesstride, grdstride, wgtstride,
                 voxel_size, absolute, membrane, bending, shears, div, niter)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_bending_jrls_')[template])
            func(np_sol, np_hes, np_grd, np_wgt,
                 shape, solstride, hesstride, grdstride, wgtstride,
                 voxel_size, absolute, membrane, bending, niter)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_lame_jrls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, membrane, shears, div, niter)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_membrane_jrls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, membrane, niter)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_flow, 'relax_absolute_jrls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, niter)

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
    nbatch = out.ndim - ndim - 1
    nc = out.shape[-1]
    if ndim > 3:
        raise ValueError('field_matvec only implemented up to dimension 3')

    np_inp = inp.numpy()
    np_out = out.numpy()

    reduce_t = np.float64
    offset_t = np.int64
    scalar_t = np_inp.dtype

    shape, instride = cinfo(np_inp, dtype=offset_t)
    _, outstride = cinfo(np_out, dtype=offset_t)

    bending = np.ascontiguousarray(bending, dtype=reduce_t)
    membrane = np.ascontiguousarray(membrane, dtype=reduce_t)
    absolute = np.ascontiguousarray(absolute, dtype=reduce_t)
    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, {nc}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if sum(bending) == sum(membrane) == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, 'matvec_bending')[template])
        func(np_out, np_inp, shape, outstride, instride,
             voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, 'matvec_membrane')[template])
        func(np_out, np_inp, shape, outstride, instride,
             voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, 'matvec_absolute')[template])
        func(np_out, np_inp, shape, outstride, instride, absolute)
    elif op == '=':
        out.zero_()

    return out


def field_matvec_rls(out, inp, wgt, bound, voxel_size,
                     absolute, membrane, bending,
                     op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, nc) tensor
    inp : (*batch, *spatial, nc) tensor
    wgt : (*batch, *spatial, nc|1) tensor
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]
    op : {'set', 'add', 'sub'}

    Returns
    -------
    out : (*batch, *spatial, channels) tensor
    """
    ndim = len(voxel_size)
    nbatch = out.ndim - ndim - 1
    nc = out.shape[-1]
    if ndim > 3:
        raise ValueError('field_matvec_rls only implemented up to dimension 3')
    joint = 'j' if wgt.shape[-1] == 1 else ''

    np_inp = inp.numpy()
    np_out = out.numpy()
    np_wgt = wgt.numpy()

    reduce_t = np.double
    offset_t = np.int64
    scalar_t = np_inp.dtype

    shape, instride = cinfo(np_inp, dtype=offset_t)
    _, outstride = cinfo(np_out, dtype=offset_t)
    _, wgtstride = cinfo(np_wgt, dtype=offset_t)

    bending = np.ascontiguousarray(bending, dtype=reduce_t)
    membrane = np.ascontiguousarray(membrane, dtype=reduce_t)
    absolute = np.ascontiguousarray(absolute, dtype=reduce_t)
    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, {nc}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if sum(bending) == sum(membrane) == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'matvec_bending_{joint}rls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'matvec_membrane_{joint}rls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'matvec_absolute_{joint}rls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             absolute)
    elif not op:
        out.zero_()

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
    nbatch = out.ndim - ndim - 1
    nc = out.shape[-1]
    if ndim > 3:
        raise ValueError('field_kernel only implemented up to dimension 3')

    np_out = out.numpy()

    reduce_t = np.double
    offset_t = np.int64
    scalar_t = np_out.dtype

    shape, stride = cinfo(np_out, dtype=offset_t)

    bending = np.ascontiguousarray(bending, dtype=reduce_t)
    membrane = np.ascontiguousarray(membrane, dtype=reduce_t)
    absolute = np.ascontiguousarray(absolute, dtype=reduce_t)
    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, {nc}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if sum(bending) == sum(membrane) == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)
    if op == '=':
        out.zero_()

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'kernel_bending')[template])
        func(np_out, shape, stride, voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'kernel_membrane')[template])
        func(np_out, shape, stride, voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'kernel_absolute')[template])
        func(np_out, shape, stride, absolute)

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
    nbatch = out.ndim - ndim - 1
    nc = out.shape[-1]
    if ndim > 3:
        raise ValueError('field_diag only implemented up to dimension 3')

    np_out = out.numpy()

    reduce_t = np.double
    offset_t = np.int64
    scalar_t = np_out.dtype

    shape, stride = cinfo(np_out, dtype=offset_t)

    bending = np.ascontiguousarray(bending, dtype=reduce_t)
    membrane = np.ascontiguousarray(membrane, dtype=reduce_t)
    absolute = np.ascontiguousarray(absolute, dtype=reduce_t)
    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, {nc}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if sum(bending) == sum(membrane) == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)
    if op == '=':
        out.zero_()

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'diag_bending')[template])
        func(np_out, shape, stride, voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'diag_membrane')[template])
        func(np_out, shape, stride, voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'diag_absolute')[template])
        func(np_out, shape, stride, absolute)

    return out


def field_diag_rls(out, wgt, bound, voxel_size,
                   absolute, membrane, bending,
                   op=''):
    """
    Parameters
    ----------
    out : (*batch, *spatial, nc) tensor
    wgt : (*batch, *spatial, nc|1) tensor
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
    nbatch = out.ndim - ndim - 1
    nc = out.shape[-1]
    if ndim > 3:
        raise ValueError('field_diag_rls only implemented up to dimension 3')
    joint = 'j' if wgt.shape[-1] == 1 else ''

    np_out = out.numpy()
    np_wgt = wgt.numpy()

    reduce_t = np.double
    offset_t = np.int64
    scalar_t = np_out.dtype

    shape, stride = cinfo(np_out, dtype=offset_t)
    _, wgtstride = cinfo(np_wgt, dtype=offset_t)

    bending = np.ascontiguousarray(bending, dtype=reduce_t)
    membrane = np.ascontiguousarray(membrane, dtype=reduce_t)
    absolute = np.ascontiguousarray(absolute, dtype=reduce_t)
    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)

    op = '+' if op in ('add', '+') else '-' if op in ('sub', '-') else '='
    template = f"{nbatch}, {ndim}, {nc}, '{op}'"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if sum(bending) == sum(membrane) == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)
    if op == '=':
        out.zero_()

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'diag_bending_{joint}rls')[template])
        func(np_out, np_wgt, shape, stride, wgtstride, voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'diag_membrane_{joint}rls')[template])
        func(np_out, np_wgt, shape, stride, wgtstride, voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'diag_absolute_{joint}rls')[template])
        func(np_out, np_wgt, shape, stride, wgtstride, absolute)

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
    sol : (*batch, *spatial, ndim) tensor
    """
    ndim = len(voxel_size)
    nbatch = sol.ndim - ndim - 1
    nc = sol.shape[-1]
    if ndim > 3:
        raise ValueError('field_relax_ only implemented up to dimension 3')

    np_sol = sol.numpy()
    np_hes = hes.numpy()
    np_grd = grd.numpy()

    offset_t = np.int64
    reduce_t = np.float64
    scalar_t = np_sol.dtype

    bending = np.ascontiguousarray(bending, dtype=reduce_t)
    membrane = np.ascontiguousarray(membrane, dtype=reduce_t)
    absolute = np.ascontiguousarray(absolute, dtype=reduce_t)
    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)

    shape, solstride = cinfo(np_sol, dtype=offset_t)
    _, hesstride = cinfo(np_hes, dtype=offset_t)
    _, grdstride = cinfo(np_grd, dtype=offset_t)

    template = f"{nbatch}, {ndim}, {nc}"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if sum(bending) == sum(membrane) == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if bending:
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, 'relax_bending_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, membrane, bending, niter)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, 'relax_membrane_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, membrane, niter)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, 'relax_absolute_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, niter)

    return sol


def field_relax_rls_(sol, hes, grd, wgt, niter, bound, voxel_size,
                     absolute, membrane, bending):
    """
    Parameters
    ----------
    sol : (*batch, *spatial, nc) tensor
    hes : (*batch, *spatial, (nc*(nc+1))//2) tensor
    grd : (*batch, *spatial, nc) tensor
    wgt : (*batch, *spatial, nc|1) tensor
    niter : int
    bound : (ndim,) list[int]
    voxel_size : (ndim,) list[float]
    absolute : (nc,) list[float]
    membrane : (nc,) list[float]
    bending : (nc,) list[float]

    Returns
    -------
    sol : (*batch, *spatial, ndim) tensor
    """
    ndim = len(voxel_size)
    nbatch = sol.ndim - ndim - 1
    nc = sol.shape[-1]
    if ndim > 3:
        raise ValueError('field_relax_rls_ only implemented up to dimension 3')
    joint = 'j' if wgt.shape[-1] == 1 else ''

    np_sol = sol.numpy()
    np_hes = hes.numpy()
    np_grd = grd.numpy()
    np_wgt = wgt.numpy()

    offset_t = np.int64
    reduce_t = np.float64
    scalar_t = np_sol.dtype

    bending = np.ascontiguousarray(bending, dtype=reduce_t)
    membrane = np.ascontiguousarray(membrane, dtype=reduce_t)
    absolute = np.ascontiguousarray(absolute, dtype=reduce_t)
    voxel_size = np.ascontiguousarray(voxel_size, dtype=reduce_t)

    shape, solstride = cinfo(np_sol, dtype=offset_t)
    _, hesstride = cinfo(np_hes, dtype=offset_t)
    _, grdstride = cinfo(np_grd, dtype=offset_t)
    _, wgtstride = cinfo(np_wgt, dtype=offset_t)

    template = f"{nbatch}, {ndim}, {nc}"
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t) + ', '
    if sum(bending) == sum(membrane) == 0:
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)

    if bending:
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'relax_bending_{joint}rls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, membrane, bending, niter)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'relax_membrane_{joint}rls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, membrane, niter)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_field, f'relax_absolute_{joint}rls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, niter)

    return sol