#ifndef JIT_DISTANCE_SPLINEDIST_H
#define JIT_DISTANCE_SPLINEDIST_H
#include "../spline.h"
#include "../bounds.h"
#include "../pushpull.h"
#include "../utils.h"


namespace jf {
namespace distance_spline {


template <int D, spline::type S, bound::type B, typename scalar_t, typename offset_t>
class SplineDist {
public:
    using PP = pushpull::PushPull<1, S, B>;
    static constexpr scalar_t gold  = static_cast<scalar_t>(1.618033988749895);
    static constexpr scalar_t igold = static_cast<scalar_t>(0.3819660112501051);
    static constexpr scalar_t tiny  = static_cast<scalar_t>(1e-8);
    static constexpr scalar_t zero  = static_cast<scalar_t>(0);
    static constexpr scalar_t one   = static_cast<scalar_t>(1);
    static constexpr offset_t ndim  = static_cast<offset_t>(D);
    static constexpr offset_t nostride = static_cast<offset_t>(1);

    __device__ static inline
    void min_table(
              scalar_t * best_time, scalar_t * best_dist, const scalar_t * loc, 
              const scalar_t * coeff, const scalar_t * time, offset_t tsize, offset_t tstride,
              offset_t lstride, offset_t csize, offset_t cstride, offset_t cstride_channel
    )
    {
        scalar_t loct[D];
        scalar_t the_best_time = zero;
        scalar_t the_best_dist = static_cast<scalar_t>(1./0.);

        for (offset_t t=0; t<tsize; ++t) 
        {
            PP::pull(loct, coeff, time + t * tstride, &csize, &cstride, 
                     ndim, nostride, cstride_channel);
            scalar_t dist = 0;
            for (offset_t d=0; d<ndim; ++d) 
            {
                loct[d] -= loc[d * lstride];
                dist += loct[d] * loct[d];
            }
            if (dist < the_best_dist) 
            {
                the_best_dist = dist;
                the_best_time = time[t * tstride];
            }
        }

        *best_time = the_best_time;
        *best_dist = the_best_dist;

    }


    __device__ static inline
    void min_brent(
              scalar_t * best_time, scalar_t * best_dist, const scalar_t * loc,
              const scalar_t * coeff,  offset_t lstride, offset_t csize, offset_t cstride, offset_t cstride_channel, 
              offset_t max_iter, scalar_t tol, scalar_t step
    )
    {
        scalar_t loct[D];
        scalar_t a, a0, a1, a2, f, f0, f1, f2, b0, b1;
        a0 = *best_time;
        f0 = *best_dist;

        // Evaluate the squared distance
        auto closure = [&](scalar_t t)
        {
            PP::pull(loct, coeff, &t, &csize, &cstride, ndim, nostride, cstride_channel);
            scalar_t dist = 0;
            for (offset_t i=0; i<ndim; ++i) 
            {
                loct[i] -= loc[i * lstride];
                dist += loct[i] * loct[i];
            }
            return dist;
        };

        // Fit a quadratic to three points (a0, f0), (a1, f1), (a2, f2)
        // and return the location of its minimum, and its quadratic factor
        //
        // inp: a0, a1, a2, f0, f1, f2
        // out: d (location), s (quadratic factor)
        auto quad_min = [&](scalar_t & d, scalar_t & s) 
        {
            scalar_t a00 = a0 * a0;
            scalar_t a11 = a1 * a1;
            scalar_t a22 = a2 * a2;
            scalar_t y01 = a00 - a11;
            scalar_t y02 = a00 - a22;
            scalar_t y12 = a11 - a22;
            scalar_t a01 = a0 - a1;
            scalar_t a02 = a0 - a2;
            scalar_t a12 = a1 - a2;
            d = 0.5 * (f0 * y12 + f1 * y02 + f2 * y01) / (f0 * a12 + f1 * a02 + f2 * a01);
            s = f0 / (y01 * y02) + f1 / (y01 * y12) + f2 / (y02 * y12);
        };

        // check progress and update bracket
        // inp: a, a0, a1, a2, f, f0, f1, f2
        // out: a0, a1, a2, f0, f1, f2
        auto update_bracket = [&]()
        {
            // f2 < f1 < f0 so (assuming unicity) the minimum is in
            // (a1, a2) or (a2, inf)

            if ((a1 < a) == (a < a2))   // a in (a1, a2)
            {
                if (f < f2)             // minimum in (a1, a2) - done
                {
                    a0 = a1;
                    a1 = a;
                    f0 = f1;
                    f1 = f;
                    return;
                }
                else if (f1 < f)        // minimum in (a0, a)
                {
                    a2 = a;
                    f2 = f;
                    return;
                }
            }
            // shift by one point
            a0 = a1;
            a1 = a2;
            a2 = a;
            f0 = f1;
            f1 = f2;
            f2 = f;
        };

        // Bracket the minimum
        //
        // Parameters
        // ----------
        // a0 : Initial parameter
        // f0 : Initial value
        //
        // Returns
        // -------
        // a0, a1, a2, f0, f1, f2
        //     (a1, f1) is the current estimate of the minimum location and value
        //     a1 is in (a0, a2) or (a2, a0)
        //     f1 is lower than both f0 and f2
        auto bracket = [&]()
        {
            a1 = a0 + step;
            f1 = closure(a1);

            // sort such that f1 < f0
            if (f1 > f0) 
            {
                swap(a0, a1);
                swap(f0, f1);
            }

            a2 = a1 + (a1 - a0) * gold;
            f2 = closure(a2);

            for (offset_t n=0; n<max_iter; ++n)
            {
                if (f1 < f2) break;

                // fit quadratic polynomial
                scalar_t delta, s;
                quad_min(delta, s);
                delta -= a1;
                if (s > 0)
                    // quadratic has a minimum
                    // -> clamp jump using golden ratio
                    a = a1 + min(delta, (1 + gold) * (a2 - a1));
                else
                    // quadratic has a maximum
                    // -> use golden ratio jump
                    a = a2 + gold * (a2 - a1);

                // evaluate new point
                f = closure(a);

                // check progress and update bracket
                update_bracket();
            }
        };

        // Sort the pairs (a0, f0), (a1, f1), (a2, f2) such that f0 < f1 < f2
        // inp/out: a0, a1, a2, f0, f1, f2
        auto search_sort = [&]()
        {
            if (f2 < f1) 
            {
                swap(a1, a2);
                swap(f1, f2);
            }
            if (f1 < f0) 
            {
                swap(a0, a1);
                swap(f0, f1);
            }
            if (f2 < f1) 
            {
                swap(a1, a2);
                swap(f1, f2);
            }
        };

        // Check whether to use bisection rather than interpolation.
        // Do not use extremum of the quadratic fit if:
        // - it is a maximum (s < 0), or
        // - jump is larger than half the second-to-last jump, or
        // - new point is too close from brackets
        //
        // s: quadratic factor of the quadratic fit
        // d: proposed jump
        // d1: jump from two iterations ago
        // tiny: tolerance when checking if proposed solution is in the bracket
        auto use_bisection = [&](scalar_t s, scalar_t d, scalar_t d1, scalar_t tiny)
        {
            return (
                (s < 0) || // quadratic fit has no minimum
                (abs(d) > abs(d1) / 2) || // jump is larger than half the second-to-last jump
                !((b0 + tiny < a0 + d) && (a0 + d < b1 - tiny)) // solution is not in brackets
            );
        };

        // inp: a0, a1, a2, f0, f1, f2, b0, b1
        // out: a, a0, a1, a2, f, f0, f1, f2, b0, b1
        auto update_search = [&]()
        {
            if (f < f0)
            {
                if (a < a0)
                    b1 = a0;
                else
                    b0 = a0;

                a2 = a1;
                a1 = a0;
                a0 = a;
                f2 = f1;
                f1 = f0;
                f0 = f;
            }
            else if (f0 < f)
            {
                if (a < a0)
                    b0 = a;
                else
                    b1 = a;
                
                if (f < f1)
                {
                    a2 = a1;
                    a1 = a;
                    f2 = f1;
                    f1 = f;

                    if (f < f2)
                    {
                        a2 = a;
                        f2 = f;
                    }
                }
            }
        };

        // Parameters
        // ----------
        // (a0, a1, a2, f0, f1, f2)
        //     Estimate (a1, f1) and bracket [(a0, f0), (a2, f2)] returned
        //     by the `bracket` function.
        //
        // Returns
        // -------
        // a0 : tensor
        //     Location of the minimum
        // f0 : tensor
        //     Value of the minimum
        auto search = [&]()
        {

            // initialise bracket
            b0 = a0; 
            b1 = a2;
            if (b1 < b0)
                swap(b0, b1);
            // sort by values
            search_sort();

            scalar_t d = static_cast<scalar_t>(1./0.);
            scalar_t d1 = d, d0 = d;
            for (offset_t n=0; n<max_iter; ++n)
            {
                if ((abs(a0 - 0.5 * (b0 + b1)) + 0.5 * (b1 - b0)) <= 2 * tol)
                {
                    // solution is close to the middle of the bracket and 
                    // the bracket is small -> done
                    a = a0;
                    f = f0;
                    break;
                }

                // d1 = delta from two iterations ago
                d1 = d0;
                d0 = d;

                // fit quadratic polynomial
                scalar_t s, d;
                quad_min(d, s);
                d -= a0;

                // if quad has a minimum -> new point = minimum      (interpolation)
                // else                  -> new point = golden ratio (bisection)
                if (use_bisection(s, d, d1, tiny * (1 + 2 * abs(a0))))
                {
                    d = (a0 > 0.5 * (b0 + b1) ? b0 - a0 : b1 - a0) * igold;
                }

                a = a0 + d;

                // evaluate new point
                f = closure(a);

                // update bracket
                update_search();
            }
        };

        bracket();
        search();
        *best_time = a0;
        *best_dist = f0;
    }


    __device__ static inline
    void min_gaussnewton(
              scalar_t * best_time, scalar_t * best_dist, const scalar_t * loc,
              const scalar_t * coeff,  offset_t lstride, offset_t csize, offset_t cstride, offset_t cstride_channel, 
              offset_t max_iter, scalar_t tol
    )
    {
        scalar_t loct[D], locg[D];
        scalar_t t = *best_time, d = *best_dist, d0 = d, t0 = t;
        scalar_t g, h, armijo = one;
        bool success;

        // Evaluate the squared distance
        // out: d
        auto eval = [&](scalar_t t)
        {
            PP::pull(loct, coeff, &t, &csize, &cstride, ndim, nostride, cstride_channel);
            d = zero;
            for (offset_t i=0; i<D; ++i) 
            {
                loct[i] -= loc[i * lstride];
                d += loct[i] * loct[i];
            }
        };

        // Evaluate the squared distance and gradient wrt position
        // out: d, g, h
        auto eval_grad = [&](scalar_t t)
        {
            PP::pull(loct, coeff, &t, &csize, &cstride, ndim, nostride, cstride_channel);
            PP::grad(locg, coeff, &t, &csize, &cstride, ndim, nostride, cstride_channel, nostride);
            d = g = h = zero;
            for (offset_t i=0; i<ndim; ++i) 
            {
                loct[i] -= loc[i * lstride];
                h += locg[i] * locg[i];
                g += locg[i] * loct[i];
                d += loct[i] * loct[i];
            }
        };

        for (offset_t n=0; n < max_iter; ++n)
        {
            eval_grad(t);
            h += tiny;
            g /= h;

            d0 = d;
            t0 = t;
            success = false;
            for (offset_t n_ls=0; n_ls < 12; ++n_ls)
            {
                t = t0 - armijo * g;
                t = (t < zero ? zero : t > csize - 1 ? csize - 1 : t);
                eval(t);
                success = d < d0;
                if (success)
                    break;
                armijo *= 0.5;
            }
            if (!success)
            {
                d = d0;
                t = t0;
                break;
            }
            armijo *= 1.5;
            if ((d0 - d) < tol * d0)
                break;
        }

        *best_time = t;
        *best_dist = d;
    }

};

} // namespace distance_spline
} // namespace jf

#endif // JIT_DISTANCE_SPLINEDIST_H