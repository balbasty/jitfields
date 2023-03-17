from .utils import cwrap
from ..common.utils import cinfo, ctypename, bound_template
import cppyy
import numpy as np
from .utils import include

include()
cppyy.include('reg_grid.hpp')
# cppyy.include('reg_field_static_3d.hpp')


def grid_vel2mom(out, inp, bound, voxel_size,
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
        raise ValueError('grid_vel2mom only implemented up to dimension 3')

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
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_all')[template])
            func(np_out, np_inp, shape, outstride, instride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_bending')[template])
            func(np_out, np_inp, shape, outstride, instride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_lame')[template])
        func(np_out, np_inp, shape, outstride, instride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_membrane')[template])
        func(np_out, np_inp, shape, outstride, instride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_absolute')[template])
        func(np_out, np_inp, shape, outstride, instride,
             voxel_size, absolute)
    elif op == '=':
        out.zero_()

    return out


def grid_vel2mom_rls(out, inp, wgt, bound, voxel_size,
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
        raise ValueError('grid_vel2mom_rls only implemented up to dimension 3')

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
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_all_jrls')[template])
            func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_bending_jrls')[template])
            func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_lame_jrls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_membrane_jrls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'vel2mom_absolute_jrls')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             voxel_size, absolute)
    elif op == '=':
        out.zero_()

    return out


def grid_kernel(out, bound, voxel_size,
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
        raise ValueError('grid_kernel only implemented up to dimension 3')

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
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'kernel_all')[template])
            func(np_out, shape, stride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'kernel_bending')[template])
            func(np_out, shape, stride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'kernel_lame')[template])
        func(np_out, shape, stride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'kernel_membrane')[template])
        func(np_out, shape, stride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'kernel_absolute')[template])
        func(np_out, shape, stride,
             voxel_size, absolute)

    return out


def grid_diag(out, bound, voxel_size,
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
        raise ValueError('grid_diag only implemented up to dimension 3')

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
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_all')[template])
            func(np_out, shape, stride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_bending')[template])
            func(np_out, shape, stride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_lame')[template])
        func(np_out, shape, stride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_membrane')[template])
        func(np_out, shape, stride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_absolute')[template])
        func(np_out, shape, stride,
             voxel_size, absolute)

    return out


def grid_diag_rls(out, wgt, bound, voxel_size,
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
        raise ValueError('vel2mom only implemented up to dimension 3')

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
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_all')[template])
            func(np_out, np_wgt, shape, stride, wgtstride,
                 voxel_size, absolute, membrane, bending, shears, div)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_bending')[template])
            func(np_out, np_wgt, shape, stride, wgtstride,
                 voxel_size, absolute, membrane, bending)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_lame')[template])
        func(np_out, np_wgt, shape, stride, wgtstride,
             voxel_size, absolute, membrane, shears, div)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_membrane')[template])
        func(np_out, np_wgt, shape, stride, wgtstride,
             voxel_size, absolute, membrane)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'diag_absolute')[template])
        func(np_out, np_wgt, shape, stride, wgtstride,
             voxel_size, absolute)

    return out


def grid_relax_(sol, hes, grd, niter, bound, voxel_size,
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
        raise ValueError('grid_relax_ only implemented up to dimension 3')

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
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_all_')[template])
            func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
                 voxel_size, absolute, membrane, bending, shears, div, niter)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_bending_')[template])
            func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
                 voxel_size, absolute, membrane, bending, niter)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_lame_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, membrane, shears, div, niter)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_membrane_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, membrane, niter)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_absolute_')[template])
        func(np_sol, np_hes, np_grd, shape, solstride, hesstride, grdstride,
             voxel_size, absolute, niter)

    return sol


def grid_relax_rls_(sol, hes, grd, wgt, niter, bound, voxel_size,
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
        raise ValueError('grid_relax_rls_ only implemented up to dimension 3')

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
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_all_jrls_')[template])
            func(np_sol, np_hes, np_grd, np_wgt,
                 shape, solstride, hesstride, grdstride, wgtstride,
                 voxel_size, absolute, membrane, bending, shears, div, niter)
        else:
            func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_bending_jrls_')[template])
            func(np_sol, np_hes, np_grd, np_wgt,
                 shape, solstride, hesstride, grdstride, wgtstride,
                 voxel_size, absolute, membrane, bending, niter)
    elif div or shears:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_lame_jrls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, membrane, shears, div, niter)
    elif membrane:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_membrane_jrls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, membrane, niter)
    elif absolute:
        func = cwrap(getattr(cppyy.gbl.jf.reg_grid, 'relax_absolute_jrls_')[template])
        func(np_sol, np_hes, np_grd, np_wgt,
             shape, solstride, hesstride, grdstride, wgtstride,
             voxel_size, absolute, niter)

    return sol


def field_vel2mom(out, inp, bound, voxel_size,
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
    if ndim > 3:
        raise ValueError('vel2mom only implemented up to dimension 3')
    nc = out.shape[-1]

    np_inp = inp.numpy()
    np_out = out.numpy()

    reduce_t = np.float64
    offset_t = np.int64
    shape, instride = cinfo(np_inp, dtype=offset_t)
    _, outstride = cinfo(np_out, dtype=offset_t)
    nalldim = int(np_inp.ndim)

    bending = np.asarray(bending, dtype=reduce_t)
    membrane = np.asarray(membrane, dtype=reduce_t)
    absolute = np.asarray(absolute, dtype=reduce_t)

    template = f'{nc}, {nalldim}, '
    if not (any(bending) or any(membrane)):
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)
    template += ', ' + ctypename(reduce_t)          # reduce_t
    template += ', ' + ctypename(np_inp.dtype)      # scalar_t
    template += ', ' + ctypename(offset_t)          # offset_t
    op = (op + '_') if op in ('add', 'sub') else ''

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}vel2mom_bending_{ndim}d')[template], 'vel2mom')
        func(np_out, np_inp, shape, outstride, instride,
             *voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}vel2mom_membrane_{ndim}d')[template], 'vel2mom')
        func(np_out, np_inp, shape, outstride, instride,
             *voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}vel2mom_absolute_{ndim}d')[template], 'vel2mom')
        func(np_out, np_inp, shape, outstride, instride, absolute)
    elif not op:
        out.zero_()

    return out


def field_vel2mom_rls(out, inp, wgt, bound, voxel_size,
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
    if ndim > 3:
        raise ValueError('vel2mom only implemented up to dimension 3')
    nc = out.shape[-1]

    joint = 'j' if wgt.shape[-1] == 1 else ''
    if joint:
        wgt = wgt.squeeze(-1)

    np_inp = inp.numpy()
    np_out = out.numpy()
    np_wgt = wgt.numpy()

    reduce_t = np.double
    offset_t = np.int64
    scalar_t = np_inp.dtype
    shape, instride = cinfo(np_inp, dtype=offset_t)
    _, outstride = cinfo(np_out, dtype=offset_t)
    _, wgtstride = cinfo(np_wgt, dtype=offset_t)
    nalldim = int(np_inp.ndim)

    bending = np.asarray(bending, dtype=reduce_t)
    membrane = np.asarray(membrane, dtype=reduce_t)
    absolute = np.asarray(absolute, dtype=reduce_t)

    template = f'{nc}, {nalldim}, '
    if not (any(bending) or any(membrane)):
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t)
    op = (op + '_') if op in ('add', 'sub') else ''

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}vel2mom_bending_{joint}rls_{ndim}d')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             *voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}vel2mom_membrane_{joint}rls_{ndim}d')[template])
        func(np_out, np_inp, np_wgt, shape, outstride, instride, wgtstride,
             *voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}vel2mom_absolute_{joint}rls_{ndim}d')[template])
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
    if ndim > 3:
        raise ValueError('vel2mom only implemented up to dimension 3')
    nc = out.shape[-1]

    np_out = out.numpy()

    reduce_t = np.double
    offset_t = np.int64
    scalar_t = np_out.dtype
    shape, stride = cinfo(np_out, dtype=offset_t)
    nalldim = int(np_out.ndim)

    bending = np.asarray(bending, dtype=reduce_t)
    membrane = np.asarray(membrane, dtype=reduce_t)
    absolute = np.asarray(absolute, dtype=reduce_t)

    template = f'{nc}, {nalldim}, '
    if not (any(bending) or any(membrane)):
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t)
    op = (op + '_') if op in ('add', 'sub') else ''
    if not op:
        out.zero_()

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}kernel_bending_{ndim}d')[template])
        func(np_out, shape, stride,
             *voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}kernel_membrane_{ndim}d')[template])
        func(np_out, shape, stride,
             *voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}kernel_absolute_{ndim}d')[template])
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
    if ndim > 3:
        raise ValueError('vel2mom only implemented up to dimension 3')
    nc = out.shape[-1]

    np_out = out.numpy()

    reduce_t = np.double
    offset_t = np.int64
    scalar_t = np_out.dtype
    shape, stride = cinfo(np_out, dtype=offset_t)
    nalldim = int(np_out.ndim)

    bending = np.asarray(bending, dtype=reduce_t)
    membrane = np.asarray(membrane, dtype=reduce_t)
    absolute = np.asarray(absolute, dtype=reduce_t)
    voxel_size = list(map(float, voxel_size))

    template = f'{nc}, {nalldim}, '
    if not (any(bending) or any(membrane)):
        template += bound_template(['dct2']*ndim)
    else:
        template += bound_template(bound)
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t)
    op = (op + '_') if op in ('add', 'sub') else ''
    if not op:
        out.zero_()

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}diag_bending_{ndim}d')[template])
        func(np_out, shape, stride,
             *voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}diag_membrane_{ndim}d')[template])
        func(np_out, shape, stride,
             *voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}diag_absolute_{ndim}d')[template])
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
    if ndim > 3:
        raise ValueError('vel2mom only implemented up to dimension 3')
    nc = out.shape[-1]

    joint = 'j' if wgt.shape[-1] == 1 else ''

    np_out = out.numpy()
    np_wgt = wgt.numpy()

    reduce_t = np.double
    offset_t = np.int64
    scalar_t = np_out.dtype
    shape, stride = cinfo(np_out, dtype=offset_t)
    _, wgtstride = cinfo(np_wgt, dtype=offset_t)
    nalldim = int(np_out.ndim)

    bending = np.asarray(bending, dtype=reduce_t)
    membrane = np.asarray(membrane, dtype=reduce_t)
    absolute = np.asarray(absolute, dtype=reduce_t)

    template = f'{nc}, {nalldim}, '
    if not (any(bending) or any(membrane)):
        template += bound_template(['nocheck']*ndim)
    else:
        template += bound_template(bound)
    template += ', ' + ctypename(reduce_t)
    template += ', ' + ctypename(scalar_t)
    template += ', ' + ctypename(offset_t)
    op = (op + '_') if op in ('add', 'sub') else ''
    if not op:
        out.zero_()

    if any(bending):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}diag_bending_{joint}rls_{ndim}d')[template])
        func(np_out, np_wgt, shape, stride, wgtstride,
             *voxel_size, absolute, membrane, bending)
    elif any(membrane):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}diag_membrane_{joint}rls_{ndim}d')[template])
        func(np_out, np_wgt, shape, stride, wgtstride,
             *voxel_size, absolute, membrane)
    elif any(absolute):
        func = cwrap(getattr(cppyy.gbl.jf.reg_field.stat, f'{op}diag_absolute_{joint}rls_{ndim}d')[template])
        func(np_out, np_wgt, shape, stride, wgtstride, absolute)

    return out
