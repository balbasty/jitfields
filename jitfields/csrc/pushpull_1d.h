/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/
#ifndef JF_PUSHPULL_1D
#define JF_PUSHPULL_1D
#include "cuda_switch.h"
#include "spline.h"
#include "bounds.h"
#include "pushpull_utils.h"

namespace jf {
namespace pushpull {

/***********************************************************************
 *
 *                               NEAREST
 *
 **********************************************************************/
template <bound::type B> struct PushPull<one, Z, B> {
    using utils = PushPullUtils<Z, B>;
    using self = PushPull<one, Z, B>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix;
        signed char fx;
        utils::index(x, nx, ix, fx);
        ix *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = bound::get(inp, ix, fx);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix;
        signed char fx;
        utils::index(x, nx, ix, fx);
        ix *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            bound::add(out, ix, *inp, fx);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t    ix;
        signed char fx;
        utils::index(x, nx, ix, fx);
        ix *= sx;

        bound::add(out, ix, 1, fx);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        self::push(out, ginp, x, nx, isx, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        self::pull(out, ginp, x, nx, sx, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, const scalar_t * inp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        *gout = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};

/***********************************************************************
 *
 *                               LINEAR
 *
 **********************************************************************/
template <bound::type B> struct PushPull<one, L, B> {
    using utils = PushPullUtils<L, B>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      static_cast<reduce_t>(bound::get(inp, x0, f0)) * w0
                    + static_cast<reduce_t>(bound::get(inp, x1, f1)) * w1);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x1, f1)
                    - bound::cget<reduce_t>(inp, x0, f0));
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            // compute input spatial gradient
            acc += gval * (bound::cget<reduce_t>(inp, x1 * isx, f1)
                         - bound::cget<reduce_t>(inp, x0 * isx, f0));
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // pull incoming gradient
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(ginp, x0, f0) * w0
                    + bound::cget<reduce_t>(ginp, x1, f1) * w1);
            // compute input spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (bound::cget<reduce_t>(ginp, x1, f1)
                        - bound::cget<reduce_t>(ginp, x0, f0));
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, const scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        // compute input spatial gradient
        *gout = static_cast<scalar_t>(bound::cget<reduce_t>(ginp, x1, f1)
                                    - bound::cget<reduce_t>(ginp, x0, f0));
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};

/***********************************************************************
 *
 *                               QUADRATIC
 *
 **********************************************************************/
template <bound::type B> struct PushPull<one, Q, B> {
    using utils = PushPullUtils<Q, B>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1, x2;
        reduce_t w0, w1, w2;
        signed char   f0, f1, f2;
        utils::index(x, nx, x0, x1, x2, w0, w1, w2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * w0
                    + bound::cget<reduce_t>(inp, x1, f1) * w1
                    + bound::cget<reduce_t>(inp, x2, f2) * w2);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1, x2;
        reduce_t    w0, w1, w2;
        signed char f0, f1, f2;
        utils::index(x, nx, x0, x1, x2, w0, w1, w2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
            bound::add(out, x2, val * w2, f2);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t x0, x1, x2;
        reduce_t w0, w1, w2;
        signed char f0, f1, f2;
        utils::index(x, nx, x0, x1, x2, w0, w1, w2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
        bound::add(out, x2, w2, f2);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1, x2;
        reduce_t    w0, w1, w2;
        reduce_t    g0, g1, g2;
        signed char f0, f1, f2;
        utils::gindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * g0
                    + bound::cget<reduce_t>(inp, x1, f1) * g1
                    + bound::cget<reduce_t>(inp, x2, f2) * g2);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t    x0, x1, x2;
        reduce_t    w0, w1, w2;
        reduce_t    g0, g1, g2;
        signed char f0, f1, f2;
        utils::gindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, f0, f1, f2);

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            bound::add(out, x2 * osx, gval * w2, f2);
            // compute input spatial gradient
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * g0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * g1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * g2);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t    x0, x1, x2;
        reduce_t    w0, w1, w2;
        reduce_t    g0, g1, g2;
        signed char f0, f1, f2;
        utils::gindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t ginp0 = bound::cget<reduce_t>(ginp, x0, f0);
            reduce_t ginp1 = bound::cget<reduce_t>(ginp, x1, f1);
            reduce_t ginp2 = bound::cget<reduce_t>(ginp, x2, f2);
            // pull incoming gradient
            *out = static_cast<scalar_t>(ginp0 * w0 + ginp1 * w1 + ginp2 * w2);
            // compute incoming gradient spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (ginp0 * g0 + ginp1 * g1 + ginp2 * g2);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, const scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t x0, x1, x2;
        reduce_t w0, w1, w2;
        reduce_t g0, g1, g2;
        signed char f0, f1, f2;
        utils::gindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        // compute input spatial gradient
        *gout = bound::cget<reduce_t>(ginp, x0, f0) * g0
              + bound::cget<reduce_t>(ginp, x1, f1) * g1
              + bound::cget<reduce_t>(ginp, x2, f2) * g2;
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t x0, x1, x2;
        reduce_t w0, w1, w2;
        reduce_t g0, g1, g2;
        reduce_t h0, h1, h2;
        signed char f0, f1, f2;
        utils::hindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, h0, h1, h2,
                      f0, f1, f2);

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, static_cast<scalar_t>(gval * g0), f0);
            bound::add(out, x1 * osx, static_cast<scalar_t>(gval * g1), f1);
            bound::add(out, x2 * osx, static_cast<scalar_t>(gval * g2), f2);
            // compute input spatial hessian
            acc += gval * (static_cast<reduce_t>(bound::get(inp, x0 * isx, f0)) * h0
                         + static_cast<reduce_t>(bound::get(inp, x1 * isx, f1)) * h1
                         + static_cast<reduce_t>(bound::get(inp, x2 * isx, f2)) * h2);
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                               CUBIC
 *
 **********************************************************************/
template <bound::type B> struct PushPull<one, C, B> {
    using utils = PushPullUtils<C, B>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        signed char   f0, f1, f2, f3;
        utils::index(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * w0
                    + bound::cget<reduce_t>(inp, x1, f1) * w1
                    + bound::cget<reduce_t>(inp, x2, f2) * w2
                    + bound::cget<reduce_t>(inp, x3, f3) * w3);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        signed char   f0, f1, f2, f3;
        utils::index(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
            bound::add(out, x2, val * w2, f2);
            bound::add(out, x3, val * w3, f3);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        signed char   f0, f1, f2, f3;
        utils::index(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
        bound::add(out, x2, w2, f2);
        bound::add(out, x3, w3, f3);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        signed char f0, f1, f2, f3;
        utils::gindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * g0
                    + bound::cget<reduce_t>(inp, x1, f1) * g1
                    + bound::cget<reduce_t>(inp, x2, f2) * g2
                    + bound::cget<reduce_t>(inp, x3, f3) * g3);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        signed char f0, f1, f2, f3;
        utils::gindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      f0, f1, f2, f3);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            bound::add(out, x2 * osx, gval * w2, f2);
            bound::add(out, x3 * osx, gval * w3, f3);
            // compute input spatial gradient
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * g0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * g1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * g2
                         + bound::cget<reduce_t>(inp, x3 * isx, f3) * g3);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        signed char f0, f1, f2, f3;
        utils::gindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t ginp0 = bound::cget<reduce_t>(ginp, x0, f0);
            reduce_t ginp1 = bound::cget<reduce_t>(ginp, x1, f1);
            reduce_t ginp2 = bound::cget<reduce_t>(ginp, x2, f2);
            reduce_t ginp3 = bound::cget<reduce_t>(ginp, x3, f3);
            // pull incoming gradient
            *out = static_cast<scalar_t>(ginp0 * w0 + ginp1 * w1 +
                                         ginp2 * w2 + ginp3 * w3);
            // compute incoming gradient spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (ginp0 * g0 + ginp1 * g1 + ginp2 * g2 + ginp3 * g3);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, const scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        signed char f0, f1, f2, f3;
        utils::gindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        // compute input spatial gradient
        *gout = bound::cget<reduce_t>(ginp, x0, f0) * g0
              + bound::cget<reduce_t>(ginp, x1, f1) * g1
              + bound::cget<reduce_t>(ginp, x2, f2) * g2
              + bound::cget<reduce_t>(ginp, x3, f3) * g3;
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        reduce_t h0, h1, h2, h3;
        signed char f0, f1, f2, f3;
        utils::hindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      h0, h1, h2, h3, f0, f1, f2, f3);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * g0, f0);
            bound::add(out, x1 * osx, gval * g1, f1);
            bound::add(out, x2 * osx, gval * g2, f2);
            bound::add(out, x3 * osx, gval * g3, f3);
            // compute input spatial hessian
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * h0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * h1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * h2
                         + bound::cget<reduce_t>(inp, x3 * isx, f3) * h3);
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                                 ANY
 *
 **********************************************************************/
template <spline::type I, bound::type B>
struct PushPull<one, I, B> {
    using utils = PushPullAnyUtils<I, B>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        offset_t length = utils::index(x, nx, ix, wx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i)
                acc += bound::cget<reduce_t>(inp, ix[i], fx[i]) * wx[i];
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        offset_t length = utils::index(x, nx, ix, wx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            for (offset_t i = 0; i <= length; ++i)
                bound::add(out, ix[i], val * wx[i], fx[i]);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        offset_t length = utils::index(x, nx, ix, wx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        for (offset_t i = 0; i <= length; ++i)
            bound::add(out, ix[i], wx[i], fx[i]);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, const scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        offset_t length = utils::gindex(x, nx, ix, wx, gx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i)
                acc += static_cast<reduce_t>(bound::get(inp, ix[i], fx[i])) * gx[i];
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        offset_t length = utils::gindex(x, nx, ix, wx, gx, fx);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t gval = static_cast<reduce_t>(*ginp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i) {
                // push incoming gradient
                bound::add(out, ix[i] * osx, gval * wx[i], fx[i]);
                // compute input spatial gradient
                acc1 += bound::cget<reduce_t>(inp, ix[i] * isx, fx[i]) * gx[i];
            }
            acc += gval * acc1;
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        offset_t length = utils::gindex(x, nx, ix, wx, gx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t val = static_cast<reduce_t>(*inp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            reduce_t acc2 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i) {
                reduce_t gval = bound::cget<reduce_t>(ginp, ix[i], fx[i]);
                // pull incoming gradient
                acc1 += gval * wx[i];
                // compute incoming gradient spatial gradient
                acc2 += gval * gx[i];
            }
            *out = static_cast<scalar_t>(acc1);
            acc += val * acc2;
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, const scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        offset_t length = utils::gindex(x, nx, ix, wx, gx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        // compute input spatial gradient
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t i = 0; i <= length; ++i)
            acc += static_cast<reduce_t>(bound::get(ginp, ix[i], fx[i])) * gx[i];
        *gout = static_cast<scalar_t>(acc);
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        reduce_t    hx[8];
        offset_t length = utils::hindex(x, nx, ix, wx, gx, hx, fx);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t gval = static_cast<reduce_t>(*ginp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i) {
                // push incoming gradient
                bound::add(out, ix[i] * osx, gval * gx[i], fx[i]);
                // compute input spatial hessian
                acc1 += bound::cget<reduce_t>(inp, ix[i] * isx, fx[i]) * hx[i];
            }
            acc += gval * acc1;
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_1D
