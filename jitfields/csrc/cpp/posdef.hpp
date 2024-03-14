#ifndef JF_POSDEF_LOOP
#define JF_POSDEF_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/posdef.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

namespace jf {
namespace posdef {

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_matvec(scalar_t * out, const scalar_t * hes, const scalar_t * inp,
                const offset_t * size,
                const offset_t * stride_out,
                const offset_t * stride_hes,
                const offset_t * stride_inp)
{
    offset_t numel = prod<nbatch>(size);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);

        utils<type::Sym, offset_t, C>::matvec(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            internal::pointer(inp + inp_offset, isc),
            static_cast<reduce_t>(0));
    }});
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_matvec_backward(
    scalar_t * out, const scalar_t * grd, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_grd,
    const offset_t * stride_inp)
{
    offset_t numel = prod<nbatch>(size);

    // offset_t nc  = size[nbatch];
    offset_t isc = stride_inp[nbatch];
    offset_t gsc = stride_grd[nbatch];
    offset_t osc = stride_out[nbatch];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t grd_offset = index2offset<nbatch>(i, size, stride_grd);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);

        utils<type::Sym, offset_t, C>::matvec_backward(
            internal::pointer(out + out_offset, osc),
            internal::pointer(grd + grd_offset, gsc),
            internal::pointer(inp + inp_offset, isc),
            static_cast<reduce_t>(0));
    }});
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_addmatvec_(scalar_t * out, const scalar_t * hes, const scalar_t * inp,
                    const offset_t * size,
                    const offset_t * stride_out,
                    const offset_t * stride_hes,
                    const offset_t * stride_inp)
{
    offset_t numel = prod<nbatch>(size);

    offset_t nc  = size[nbatch];
    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);

        utils<type::Sym, offset_t, C>::addmatvec_(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            internal::pointer(inp + inp_offset, isc),
            static_cast<reduce_t>(0));
    }});
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_submatvec_(scalar_t * out, const scalar_t * hes, const scalar_t * inp,
                    const offset_t * size,
                    const offset_t * stride_out,
                    const offset_t * stride_hes,
                    const offset_t * stride_inp)
{
    offset_t numel = prod<nbatch>(size);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);

        utils<type::Sym, offset_t, C>::submatvec_(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            internal::pointer(inp + inp_offset, isc),
            static_cast<reduce_t>(0));
    }});
}


template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve(scalar_t * out, const scalar_t * inp,
               const scalar_t * hes, const scalar_t * wgt,
               const offset_t * size,
               const offset_t * stride_out,
               const offset_t * stride_inp,
               const offset_t * stride_hes,
               const offset_t * stride_wgt)
{
    offset_t numel = prod<nbatch>(size);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t wsc = stride_wgt ? stride_wgt[nbatch] : 0;
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t buffer[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t wgt_offset = stride_wgt ? index2offset<nbatch>(i, size, stride_wgt) : 0;

        utils<type::Sym, offset_t, C>::solve(
            internal::pointer(out + out_offset, osc),
            internal::pointer(inp + inp_offset, isc),
            internal::pointer(hes + hes_offset, hsc),
            wgt ? internal::pointer(wgt + wgt_offset, wsc) : nullptr,
            buffer, static_cast<reduce_t>(0));
    }});
}


template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve_(scalar_t * out,
                const scalar_t * hes, const scalar_t * wgt,
                const offset_t * size,
                const offset_t * stride_out,
                const offset_t * stride_hes,
                const offset_t * stride_wgt)
{
    offset_t numel = prod<nbatch>(size);

    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t wsc = stride_wgt ? stride_wgt[nbatch] : 0;
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t buffer[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t wgt_offset = stride_wgt ? index2offset<nbatch>(i, size, stride_wgt) : 0;

        utils<type::Sym, offset_t, C>::solve_(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            wgt ? internal::pointer(wgt + wgt_offset, wsc) : nullptr,
            buffer, static_cast<reduce_t>(0));
    }});
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert(scalar_t * out, const scalar_t * hes,
                const offset_t * size,
                const offset_t * stride_out,
                const offset_t * stride_hes)
{
    offset_t numel = prod<nbatch>(size);

    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t buffer[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);

        utils<type::Sym, offset_t, C>::invert(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            buffer, static_cast<reduce_t>(0));
    }});
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert_(scalar_t * hes,
                 const offset_t * size,
                 const offset_t * stride)
{
    offset_t numel = prod<nbatch>(size);

    offset_t sc = stride[nbatch];
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t buffer[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);

        utils<type::Sym, offset_t, C>::invert_(
            internal::pointer(hes + offset, sc),
            buffer, static_cast<reduce_t>(0));
    }});
}


} // namespace posdef
} // namespace jf

#endif // JF_POSDEF_LOOP
