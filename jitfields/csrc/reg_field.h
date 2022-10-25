#ifndef JF_REGULARISERS
#define JF_REGULARISERS
#include "cuda_swap.h"
#include "bounds.h"
#include "hessian.h"
#include "utils.h"

namespace jf {
namespace reg_field {

const bound::type B0 = bound::type::NoCheck;
const int zero  = 0;
const int one   = 1;
const int two   = 2;
const int three = 3;

template <int D, hessian::type H,
          bound::type BX=N0, bound::type BY=BX, bound::type BZ=BY>
class RegField {};

template <hessian::type H, bound::type BX>
class RegField<one, H, BX> {
    h_utils = hessian::utils<H>;
    bound_utils = bound::utils<BX>;

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_absolute(scalar_t * out,
                          const scalar_t * hes, const scalar_t * inp,
                          offset_t x, offset_t nx, offset_t sx,
                          offset_t nc, offset_t osc, offset_t isc, offset_t hsc,
                          const reduce_t * absolute)
    {
        const reduce_t *a = absolute;
        for (offset_t c = 0; c < nc; ++c, inp += isc, out += osc)
            *out = static_cast<scalar_t>( (*(a++)) * (*inp) );

        inp -= nc*isc;
        out -= nc*osc;
        h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_membrane(scalar_t * out,
                          const scalar_t * hes, const scalar_t * inp,
                          offset_t x, offset_t nx, offset_t sx,
                          offset_t nc, offset_t osc, offset_t isc, offset_t hsc,
                          const reduce_t * absolute, const reduce_t * membrane,
                          const reduce_t m100)
    {
        signed char f0 = bound_utils::sign(x-1, nx);
        signed char f1 = bound_utils::sign(x+1, nx);
        offset_t    x0 = (bound_utils::index(x-1, nx) - x) * sx;
        offset_t    x1 = (bound_utils::index(x+1, nx) - x) * sx;

        const reduce_t *a = absolute, *m = membrane;

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc)
        {
            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            *out = static_cast<scalar_t>(
                  (*(a++)) * center
                + (*(m++)) * (m100*(get(x0, f0) + get(x1, f1)) )
            );
        }

        inp -= nc*isc;
        out -= nc*osc;
        h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_bending(scalar_t * out,
                          const scalar_t * hes, const scalar_t * inp,
                          offset_t x, offset_t nx, offset_t sx,
                          offset_t nc, offset_t osc, offset_t isc, offset_t hsc,
                          const reduce_t * absolute, const reduce_t * membrane,
                          const reduce_t * bending,
                          const reduce_t m100, const reduce_t b100, const reduce_t b200)
    {
        signed char f00 = bound_utils::sign(x-2, nx);
        signed char f0  = bound_utils::sign(x-1, nx);
        signed char f1  = bound_utils::sign(x+1, nx);
        signed char f11 = bound_utils::sign(x+2, nx);
        offset_t    x00 = (bound_utils::index(x-2, nx) - x) * sx;
        offset_t    x0  = (bound_utils::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils::index(x+1, nx) - x) * sx;
        offset_t    x11 = (bound_utils::index(x+2, nx) - x) * sx;

        const reduce_t *a = absolute, *m = membrane, *b = bending;
        reduce_t aa, mm, bb, w100, w200;

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc)
        {
            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            aa = *(a++);
            mm = *(m++);
            bb = *(b++);
            w100 = bb * b100 + mm * m100;
            w200 = bb * b200;

            *out = static_cast<scalar_t>(
                  aa   * center
                + w100 * (get(x0, f0) + get(x1, f1))
                + w200 * (get(x00, f00) + get(x11, f11))
            );
        }

        inp -= nc*isc;
        out -= nc*osc;
        h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

} // namespace reg_field
} // namespace jf

#endif // JF_REGULARISERS
