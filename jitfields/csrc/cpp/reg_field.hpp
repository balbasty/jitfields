#ifndef JF_REGFIELD_HPP
#define JF_REGFIELD_HPP
#include "../lib/cuda_switch.h"
#include "../lib/bounds.h"
#include "../lib/batch.h"

namespace jf {
namespace reg_field {

/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/

template <bound::type BX,
          typename reduce_t, typename scalar_t, typename offset_t>
void addreg1d(scalar_t * out,  const scalar_t * inp,
              const offset_t * size,
              const offset_t * size_splinc,
              const offset_t * stride_out,
              const offset_t * stride_inp,
              const offset_t * stride_grid)
{
    offset_t numel = prod(size_grid, ndim-1);  // no outer loop across channels

    offset_t nx  = size_splinc[ndim-2];
    offset_t nc  = size_splinc[ndim-1];
    offset_t isx = stride_inp[ndim-2];
    offset_t isc = stride_inp[ndim-1];
    offset_t osc = stride_out[ndim-1];

    for (offset_t i=0; i < numel; ++i)
    {
        offset_t out_offset = index2offset(i, ndim-1, size_grid, stride_out);
        offset_t grid_offset = index2offset(i, ndim-1, size_grid, stride_grid);
        reduce_t x = static_cast<reduce_t>(grid[grid_offset]);
        if (!InFOV<extrapolate, one>::infov(x, nx)) {
            for (offset_t c=0; c<nc; ++c)
                out[out_offset + c * osc] = static_cast<scalar_t>(0);  // NaN?
            continue;
        }
        offset_t inp_offset = index2offset(i, ndim-2, size_grid, stride_inp);

        PushPull<one, IX, BX>::pull(out + out_offset, inp + inp_offset,
                                    x, nx, isx, nc, osc, isc);
    }
}


} // namespace reg_field
} // namespace jf

#endif // JF_REGFIELD_HPP
