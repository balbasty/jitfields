#include "../lib/cuda_switch.h"
#include "../lib/bounds.h"
#include "../lib/utils.h"
#include "../lib/batch.h"
#include "../lib/regularisers/field.h"
#include "../lib/posdef.h"

using namespace std;
using namespace jf;
using namespace jf::reg_field;

//======================================================================
//                              ABSOLUTE
//======================================================================

// --- ABSOLUTE: matvec -----------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_absolute(
    scalar_t * out,                  // (*batch, *spatial, channels) tensor
    const scalar_t * inp,            // (*batch, *spatial, channels) tensor
    const offset_t * _size,          // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * _stride_inp,    // [*batch, *spatial, channels] vector
    const reduce_t absolute[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride_out [nall+1]; fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp [nall+1]; fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    // compute kernel
    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t inp_offset = index2offset<nall>(i, size, stride_inp);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        Impl::template matvec_absolute<opfunc>(
            out + out_offset, inp + inp_offset, osc, isc, kernel);
    }
}

// --- ABSOLUTE: kernel ------------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void kernel_absolute(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t absolute[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride, _stride);
    offset_t sc = stride[nall];
    offset_t numel = prod<nbatch>(size);  // loop across batch only

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);

    offset_t offset = center_offset<ndim>(size+nbatch, stride+nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        Impl::template kernel_absolute<opfunc>(out + out_offset, sc, kernel);
    }
}

// --- ABSOLUTE: diagonal ----------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_absolute(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t absolute[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride, _stride);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride, loc);

        Impl::template diag_absolute<opfunc>(out + out_offset, sc, kernel);
    }
}

//======================================================================
//                              MEMBRANE
//======================================================================

// --- MEMBRANE: matvec -----------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_membrane(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * inp,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride_out [nall+1]; fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp [nall+1]; fillfrom<nall+1>(stride_inp, _stride_inp);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        Impl::template matvec_membrane<opfunc>(
            out + out_offset, inp + inp_offset,
            loc, size + nbatch, stride_inp + nbatch, osc, isc, kernel);
    }
}

// --- MEMBRANE: kernel ------------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void kernel_membrane(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nbatch>(size);

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_fullkernel_membrane(kernel, absolute, membrane, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        Impl::template kernel_membrane<opfunc>(
            out + out_offset, sc, stride + nbatch, kernel);
    }
}

// --- MEMBRANE: diagonal ----------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_membrane(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride, loc);

        Impl::template diag_membrane<opfunc>(
            out + out_offset, sc, loc, size + nbatch, kernel);
    }
}

// --- MEMBRANE: relax -------------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void relax_membrane_(
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    int niter)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride_sol [nall+1]; fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes [nall+1]; fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd [nall+1]; fillfrom<nall+1>(stride_grd, _stride_grd);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch1<ndim>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        Impl::template matvec_membrane<isub>(
            val, sol + sol_offset,
            loc, size + nbatch, stride_sol + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        Impl::template diag_membrane<set>(
            diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                              BENDING
//======================================================================

// --- BENDING: matvec ------------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_bending(
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride_out [nall+1]; fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp [nall+1]; fillfrom<nall+1>(stride_inp, _stride_inp);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        Impl::template matvec_bending<opfunc>(
            out + out_offset, inp + inp_offset,
            loc, size + nbatch, stride_inp + nbatch, osc, isc, kernel);
    }
}

// --- BENDING: kernel -------------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void kernel_bending(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride[nall+1];      fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nbatch>(size);

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_fullkernel_bending(kernel, absolute, membrane, bending, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        Impl::template kernel_bending<opfunc>(
            out + out_offset, sc, stride + nbatch, kernel);
    }
}

// --- BENDING: diagonal -----------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_bending(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride[nall+1];      fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride, loc);

        Impl::template diag_bending<opfunc>(
            out + out_offset, sc, loc, size + nbatch, kernel);
    }
}

// --- BENDING: relax --------------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void relax_bending_(
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C],
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch3<ndim>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        Impl::template matvec_bending<isub>(
            val, sol + sol_offset,
            loc, size + nbatch, stride_sol + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        Impl::template diag_bending<set>(
            diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           ABSOLUTE RLS
//======================================================================

// --- ABSOLUTE+RLS: matvec --------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_absolute_rls(
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t inp_offset = index2offset<nall>(i, size, stride_inp);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template matvec_absolute_rls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            osc, isc, wsc, kernel);
    }
}

// --- ABSOLUTE+RLS: diagonal ------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_absolute_rls(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    offset_t osc = stride_out[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template diag_absolute_rls<opfunc>(
            out + out_offset, wgt + wgt_offset, osc, wsc, kernel);
    }
}

// --- ABSOLUTE+RLS: relax ---------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void relax_absolute_rls_(
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * absolute,
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset<nall>(i, size, stride_sol);
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        Impl::template matvec_absolute_rls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            static_cast<offset_t>(1), osc, wsc, kernel);

        // diagonal
        Impl::template diag_absolute_rls<set>(
            diag, wgt + wgt_offset,
            static_cast<offset_t>(1), wsc, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           ABSOLUTE JRLS
//======================================================================

// --- ABSOLUTE+JRLS: matvec -------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_absolute_jrls(
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t inp_offset = index2offset<nall>(i, size, stride_inp);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template matvec_absolute_jrls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            osc, isc, kernel);
    }
}

// --- ABSOLUTE+JRLS: diagonal -----------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_absolute_jrls(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    offset_t osc = stride_out[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template diag_absolute_jrls<opfunc>(
            out + out_offset, wgt + wgt_offset, osc, kernel);
    }
}

// --- ABSOLUTE+JRLS: relax --------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void relax_absolute_jrls_(
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * absolute,
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset<nall>(i, size, stride_sol);
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        Impl::template matvec_absolute_jrls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        Impl::template diag_absolute_jrls<set>(
            diag, wgt + wgt_offset,
            static_cast<offset_t>(1), kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           MEMBRANE RLS
//======================================================================

// --- MEMBRANE+RLS: matvec --------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_membrane_rls(
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template matvec_membrane_rls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, wsc, kernel);
    }
}

// --- MEMBRANE+RLS: diagonal ------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_membrane_rls(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template diag_membrane_rls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, wsc, kernel);
    }
}

// --- MEMBRANE+RLS: relax ---------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void relax_membrane_rls_(
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch1<ndim>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        Impl::template matvec_membrane_rls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, wsc, kernel);

        // diagonal
        Impl::template diag_membrane_rls<set>(
            diag, wgt + wgt_offset, loc,
            size + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), wsc, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           MEMBRANE JRLS
//======================================================================

// --- MEMBRANE+JRLS: matvec ------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_membrane_jrls(
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template matvec_membrane_jrls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, kernel);
    }
}

// --- MEMBRANE+JRLS: diagonal -----------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_membrane_jrls(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template diag_membrane_jrls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, kernel);
    }
}

// --- MEMBRANE+JRLS: relax --------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void relax_membrane_jrls_(
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch1<ndim>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        Impl::template matvec_membrane_jrls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        Impl::template diag_membrane_jrls<set>(
            diag, wgt + wgt_offset, loc,
            size + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           BENDING RLS
//======================================================================

// --- BENDING+RLS: matvec ---------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_bending_rls(
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template matvec_bending_rls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, wsc, kernel);
    }
}

// --- BENDING+RLS: diagonal -------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_bending_rls(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template diag_bending_rls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, wsc, kernel);
    }
}

// --- BENDING+RLS: relax ----------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void relax_bending_rls_(
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch3<ndim>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        Impl::template matvec_bending_rls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, wsc, kernel);

        // diagonal
        Impl::template diag_bending_rls<set>(
            diag, wgt + wgt_offset, loc,
            size + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), wsc, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           BENDING JRLS
//======================================================================

// --- BENDING+JRLS: matvec --------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void matvec_bending_jrls(
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template matvec_bending_jrls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, kernel);
    }
}

// --- BENDING+JRLS: diagonal ------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void diag_bending_jrls(
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        Impl::template diag_bending_jrls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, kernel);
    }
}

// --- BENDING+JRLS: relax ---------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
__global__
void relax_bending_jrls_(
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch3<ndim>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        Impl::template matvec_bending_jrls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        Impl::template diag_bending_jrls<set>(
            diag, wgt + wgt_offset, loc,
            size + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}
