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

enum class type : char {
    None,
    L2,
    L1,
    J1,
};

template <int D, hessian::type H,
          type ABS=type::None, type MEM = type::None, type BEN = type::None,
          bound::type BX=N0, bound::type BY=BX, bound::type BZ=BY>
class RegField {};

template <int D, type T=type::None,
          bound::type BX=N0, bound::type BY=BX, bound::type BZ=BY>
class RegFieldAbsolute {};

template <int D, type T=type::None,
          bound::type BX=N0, bound::type BY=BX, bound::type BZ=BY>
class RegFieldMembrane {};

template <int D, type T=type::None,
          bound::type BX=N0, bound::type BY=BX, bound::type BZ=BY>
class RegFieldBending {};

template <int D,
          bound::type BX=N0, bound::type BY=BX, bound::type BZ=BY>
class RegFieldBase {};

/***********************************************************************
 *                      Generic regulariser fusion
 **********************************************************************/

#define JF_SIGN_VEL2MOM_1D \
    template <typename  reduce_t, typename scalar_t, typename offset_t> \
    void vel2mom( \
        /* pointers: output / hessian / input / weight maps */ \
        scalar_t * out, const scalar_t * hes, const scalar_t * inp, \
        const scalar_t * awgt, const scalar_t * mwgt, const scalar_t * bwgt, \
        /* strides */ \
        offset_t x, offset_t nx, offset_t isx, offset_t asx, offset_t msx, offset_t bsx, \
        offset_t nc, offset_t osc, offset_t isc, offset_t hsc, \
        offset_t asc, offset_t msc, offset_t bsc, \
        /* global modulation factors */ \
        const reduce_t * absolute = null, \
        const reduce_t * membrane = null, \
        const reduce_t * bending = null, \
        /* precomputed kernel elements */ \
        reduce_t m100 = static_cast<reduce_t>(0), \
        reduce_t b100 = static_cast<reduce_t>(0), \
        reduce_t b200 = static_cast<reduce_t>(0))


template <hessian::type H, bound::type BX>
class RegField<one, H, type::None, type::None, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

template <hessian::type H, bound::type BX>
class RegField<one, H, type::L2, type::None, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        RegFieldBase<one, H, BX>::vel2mom_absolute(
            out, inp, x, nx, isx, nc, osc, isc, absolute);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

template <hessian::type H, bound::type BX>
class RegField<one, H, type::None, type::L2, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return RegFieldBase<one, H, BX>::vel2mom_membrane(
            out, inp, x, nx, isx, nc, osc, isc,
            absolute, membrane, m100);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

template <hessian::type H, bound::type BX>
class RegField<one, H, type::L2, type::L2, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return RegFieldBase<one, H, BX>::vel2mom_membrane(
            out, inp, x, nx, isx, nc, osc, isc,
            absolute, membrane, m100);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

template <hessian::type H, bound::type BX>
class RegField<one, H, type::None, type::None, type::L2, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return RegFieldBase<one, H, BX>::vel2mom_bending(
            out, inp, x, nx, isx, nc, osc, isc,
            absolute, membrane, bending, m100, b100, b200);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

template <hessian::type H, bound::type BX>
class RegField<one, H, type::None, type::L2, type::L2, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return RegFieldBase<one, H, BX>::vel2mom_bending(
            out, inp, x, nx, isx, nc, osc, isc,
            absolute, membrane, bending, m100, b100, b200);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

template <hessian::type H, bound::type BX>
class RegField<one, H, type::L2, type::None, type::L2, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return RegFieldBase<one, H, BX>::vel2mom_bending(
            out, inp, x, nx, isx, nc, osc, isc,
            absolute, membrane, bending, m100, b100, b200);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

template <hessian::type H, bound::type BX>
class RegField<one, H, type::L2, type::L2, type::L2, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return RegFieldBase<one, H, BX>::vel2mom_bending(
            out, inp, x, nx, isx, nc, osc, isc,
            absolute, membrane, bending, m100, b100, b200);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};


// ABSOLUTE RLS
template <hessian::type H, bound::type BX>
class RegField<one, H, type::L1, type::None, type::None, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return vel2mom_absolute_rls(out, inp, awgt, x, nx, isx, asx,
                                    nc, osc, isc, wsc, absolute);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

// ABSOLUTE RLS (JOINT)
template <hessian::type H, bound::type BX>
class RegField<one, H, type::L1, type::None, type::None, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return vel2mom_absolute_jrls(out, inp, awgt, x, nx, isx, asx,
                                     nc, osc, isc, wsc, absolute);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

// MEMBRANE RLS
template <hessian::type H, bound::type BX>
class RegField<one, H, type::None, type::L1, type::None, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        *out = static_cast<scalar_t>(0);
        return vel2mom_absolute_rls(out, inp, awgt, x, nx, isx, asx,
                                    nc, osc, isc, wsc, absolute);
        if (hes) h_utils::addmatvec_(nc, out, osc, hes, hsc, inp, isc);
    }
};

/***********************************************************************
 *                     Mapping from impl to generic
 **********************************************************************/

template <bound::type BX>
class RegFieldAbsolute<one, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {}
};

template <bound::type BX>
class RegFieldMembrane<one, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {}
};

template <bound::type BX>
class RegFieldBending<one, type::None, BX> {
    JF_SIGN_VEL2MOM_1D
    {}
};

// ABSOLUTE L2
template <bound::type BX>
class RegFieldAbsolute<one, type::L2, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        RegFieldBase<one, BX>::vel2mom_absolute(
            out, inp, x, nx, isx, nc, osc, isc, absolute);
    }
};

// MEMBRANE L2
template <bound::type BX>
class RegFieldMembrane<one, type::L2, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        RegFieldBase<one, BX>::vel2mom_membrane(
            out, inp, x, nx, isx, nc, osc, isc,
            absolute, membrane, m100);
    }
};

// BENDING L2
template <bound::type BX>
class RegFieldBending<one, type::L2, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        RegFieldBase<one, BX>::vel2mom_bending(
            out, inp, x, nx, isx, nc, osc, isc,
            absolute, membrane, bending, m100, b100, b200);
    }
};

// ABSOLUTE RLS
template <bound::type BX>
class RegFieldAbsolute<one, type::L1, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        RegFieldBase<one, BX>::vel2mom_absolute_rls(
            out, inp, awgt, x, nx, isx, asx,
            nc, osc, isc, asc, absolute);
    }
};

// ABSOLUTE RLS (JOINT)
template <bound::type BX>
class RegFieldAbsolute<one, type::J1, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        vel2mom_absolute_jrls(
            out, inp, awgt, x, nx, isx, asx,
            nc, osc, isc, asc, absolute);
    }
};

// MEMBRANE RLS
template <bound::type BX>
class RegFieldMembrane<one, type::L1, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        vel2mom_absolute_rls(
            out, inp, mwgt, x, nx, isx, msx,
            nc, osc, isc, msc, membrane);
    }
};

// MEMBRANE JRLS
template <bound::type BX>
class RegFieldMembrane<one, type::J1, BX> {
    JF_SIGN_VEL2MOM_1D
    {
        vel2mom_absolute_jrls(
            out, inp, mwgt, x, nx, isx, msx,
            nc, osc, isc, msc, membrane);
    }
};

/***********************************************************************
 *                    Components implementation
 **********************************************************************/

template <bound::type BX>
class RegFieldBase<one, BX> {
    bound_utils = bound::utils<BX>;

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_absolute(scalar_t * out, const scalar_t * inp,
                          offset_t nc, offset_t osc, offset_t isc,
                          const reduce_t * absolute)
    {
        const reduce_t *a = absolute;
        for (offset_t c = 0; c < nc; ++c, inp += isc, out += osc)
            *out += static_cast<scalar_t>( (*(a++)) * (*inp) );
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_membrane(scalar_t * out, const scalar_t * inp,
                          offset_t x, offset_t nx, offset_t sx,
                          offset_t nc, offset_t osc, offset_t isc,
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

            reduce_t o000 = (*(m++)) * (m100*(get(x0, f0) + get(x1, f1));
            if (absolute)
                o000 += (*(a++)) * center;
            *out += static_cast<scalar_t>(o000);
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_bending(scalar_t * out, const scalar_t * inp,
                         offset_t x, offset_t nx, offset_t sx,
                         offset_t nc, offset_t osc, offset_t isc,
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

            aa = absolute ? *(a++) : static_cast<reduce_t>(0);
            mm = membrane ? *(m++) : static_cast<reduce_t>(0);
            bb = bending  ? *(b++) : static_cast<reduce_t>(0);
            w100 = bb * b100 + mm * m100;
            w200 = bb * b200;

            *out += static_cast<scalar_t>(
                  aa   * center
                + w100 * (get(x0, f0) + get(x1, f1))
                + w200 * (get(x00, f00) + get(x11, f11))
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_absolute_rls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t isx, offset_t wsx,
        offset_t nc, offset_t osc, offset_t isc, offset_t wsc,
        const reduce_t * absolute)
    {
        const reduce_t *a = absolute;
        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc, wgt += wsc)
        {
            *out += static_cast<scalar_t>(
                (*(a++)) * static_cast<reduce_t>(*wgt) * static_cast<reduce_t>(*inp)
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_absolute_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t isx, offset_t wsx,
        offset_t nc, offset_t osc, offset_t isc, offset_t wsc,
        const reduce_t * absolute)
    {
        const reduce_t *a = absolute;
        reduce_t w000 = static_cast<reduce_t>(*wgt);
        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc, wgt += wsc)
        {
            *out += static_cast<scalar_t>(
                (*(a++)) * w000 * static_cast<reduce_t>(*inp)
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_membrane_rls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t isx, offset_t wsx,
        offset_t nc, offset_t osc, offset_t isc, offset_t wsc,
        const reduce_t * membrane, const reduce_t m100)
    {
        signed char f0 = bound_utils::sign(x-1, nx);
        signed char f1 = bound_utils::sign(x+1, nx);
        offset_t    ix0 = (bound_utils::index(x-1, nx) - x);
        offset_t    ix1 = (bound_utils::index(x+1, nx) - x);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        ix0 *= isx;
        ix1 *= isx;

        const reduce_t *m = membrane;

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc, wgt += wsc)
        {

            reduce_t wcenter = static_cast<reduce_t>(*wgt);
            auto wget = [wcenter, wgt](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(wgt, o, f) + wcenter;
            };

            reduce_t w1m00 = m100 * wget(wx0, f0);
            reduce_t w1p00 = m100 * wget(wx1, f1);

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            *out += static_cast<scalar_t>(
                (*(m++)) * 0.5 * (w1m00*get(ix0, f0) + w1p00 * get(ix1, f1))
            );
        }
    }
};

} // namespace reg_field
} // namespace jf

#endif // JF_REGULARISERS
