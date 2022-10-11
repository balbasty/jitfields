// Compute spline interpolating coefficients
//
// These functions are ported from the C routines in SPM's bsplines.c
// by John Ashburner, which are themselves ports from Philippe Thevenaz's
// code. JA furthermore derived the initial conditions for the DFT ("wrap around")
// boundary conditions.
//
// Note that similar routines are available in scipy with boundary conditions
// DCT1 ("mirror"), DCT2 ("reflect") and DFT ("wrap"); all derived by P. Thevenaz,
// according to the comments. Our DCT2 boundary conditions are ported from
// scipy.
//
// Only boundary conditions DCT1, DCT2 and DFT are implemented.
//
// References
// ----------
// ..[1]  M. Unser, A. Aldroubi and M. Eden.
//        "B-Spline Signal Processing: Part I-Theory,"
//        IEEE Transactions on Signal Processing 41(2):821-832 (1993).
// ..[2]  M. Unser, A. Aldroubi and M. Eden.
//        "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
//        IEEE Transactions on Signal Processing 41(2):834-848 (1993).
// ..[3]  M. Unser.
//        "Splines: A Perfect Fit for Signal and Image Processing,"
//        IEEE Signal Processing Magazine 16(6):22-38 (1999).

#ifndef JF_SPLINC
#define JF_SPLINC
#include "cuda_switch.h"
#include "spline.h"
#include "bounds.h"
#include "utils.h"

namespace jf {
namespace splinc {

namespace _splinc {

template <typename scalar_t, typename reduce_t, typename offset_t>
inline __device__ scalar_t dft_initial(scalar_t * inp, reduce_t pole,
                                       offset_t size, offset_t stride)
{
    offset_t max_iter = static_cast<offset_t>(ceil(-30./log(fabs(pole))));
    max_iter = min(max_iter, size);

    reduce_t out = static_cast<reduce_t>(inp[0]);
    reduce_t polen = pole;
    inp += (size - 1) * stride;
    for (offset_t i = 1; i < max_iter; ++i, inp -= stride) {
        out   += polen * static_cast<reduce_t>(*inp);
        polen *= pole;
    }
    out /= (1. - polen);
    return static_cast<scalar_t>(out);
}

template <typename scalar_t, typename reduce_t, typename offset_t>
inline __device__ scalar_t dft_final(scalar_t * inp, reduce_t pole,
                                     offset_t size, offset_t stride)
{
    offset_t max_iter = static_cast<offset_t>(ceil(-30./log(fabs(pole))));
    max_iter = min(max_iter, size);

    reduce_t out = static_cast<reduce_t>(inp[(size-1) * stride]) * pole;
    reduce_t polen = pole;
    for (offset_t i = 0; i < max_iter-1; ++i, inp += stride) {
        polen *= pole;
        out   += polen * static_cast<reduce_t>(*inp);
    }
    out /= polen - 1.;
    return static_cast<scalar_t>(out);
}

template <typename scalar_t, typename reduce_t, typename offset_t>
inline __device__ scalar_t dct1_initial(scalar_t * inp, reduce_t pole,
                                        offset_t size, offset_t stride)
{
    offset_t max_iter = static_cast<offset_t>(ceil(-30./log(fabs(pole))));
    max_iter = min(max_iter, size);

    reduce_t out;
    if (max_iter < size)
    {
        reduce_t polen = pole;
        out = static_cast<reduce_t>(inp[0]);
        inp += stride;
        for (offset_t i = 1; i < max_iter; ++i, inp += stride)
        {
            out   += polen * static_cast<reduce_t>(*inp);
            polen *= pole;
        }
    }
    else
    {
        reduce_t polen  = pole;
        reduce_t ipole  = 1./pole;
        reduce_t ipolen = pow(pole, static_cast<reduce_t>(size) - 1.);
        out = static_cast<reduce_t>(inp[0])
            + static_cast<reduce_t>(inp[(size - 1)*stride]) * ipolen;
        ipolen *= ipolen * ipole;
        inp += stride;
        for (offset_t i = 1; i < size-1; ++i, inp += stride)
        {
            out    += (polen + ipolen) * static_cast<reduce_t>(*inp);
            polen  *= pole;
            ipolen *= ipole;
        }
        out /= (1. - ipole*ipole);
    }
    return static_cast<scalar_t>(out);
}

template <typename scalar_t, typename reduce_t, typename offset_t>
inline __device__ scalar_t dct1_final(scalar_t * inp, reduce_t pole,
                                      offset_t size, offset_t stride)
{
    inp += (size - 1) * stride;
    reduce_t out = static_cast<reduce_t>(*inp);
    inp -= stride;
    out += pole * static_cast<reduce_t>(*inp);
    out *= pole / (pole*pole - 1.);
    return static_cast<scalar_t>(out);
}


template <typename scalar_t, typename reduce_t, typename offset_t>
inline __device__ reduce_t dct2_initial(scalar_t * inp, reduce_t pole,
                                        offset_t size, offset_t stride)
{
// Ported from scipy:
// https://github.com/scipy/scipy/blob/master/scipy/ndimage/src/ni_splines.c
//
// I (YB) unwarped and simplied the terms
//
// It should certainly be possible to derive a version for max_iter < n,
// as JA did for DCT1, to avoid long recursions when `n` is large. But
// I think it would require a more complicated anticausal/final condition.

    reduce_t polen = pole;
    reduce_t polen_last = pow(pole, static_cast<reduce_t>(size));
    scalar_t * inp_last = inp + (size - 1) * stride;
    scalar_t inp0 = static_cast<reduce_t>(*inp);
    reduce_t out = inp0  + static_cast<reduce_t>(*inp_last) * polen_last;
    inp += stride;
    inp_last -= stride;
    for (offset_t i=1; i<size; ++i, inp += stride, inp_last -= stride ) {
        out += polen * (static_cast<reduce_t>(*inp) +
                        static_cast<reduce_t>(*inp_last) * polen_last);
        polen *= pole;
    }

    out *= pole / (1. - polen * polen);
    out += inp0;
    return static_cast<scalar_t>(out);
}

template <typename scalar_t, typename reduce_t, typename offset_t>
inline __device__ scalar_t dct2_final(scalar_t * inp, reduce_t pole,
                                      offset_t size, offset_t stride)
{
    reduce_t out = static_cast<reduce_t>(inp[(size - 1) * stride]);
    out *= pole / (pole - 1.);
    return static_cast<scalar_t>(out);
}

} // namespace _splinc

template <typename scalar_t>
inline __device__ int get_poles(int order, scalar_t * poles)
{
    switch (order) {
        case 0:
        case 1:
            return 0;
        case 2:
            poles[0] = static_cast<scalar_t>(sqrt(8.) - 3.);
            return 1;
        case 3:
            poles[0] = static_cast<scalar_t>(sqrt(3.) - 2.);
            return 1;
        case 4:
            poles[0] = static_cast<scalar_t>(sqrt(664. - sqrt(438976.)) + sqrt(304.) - 19.);
            poles[1] = static_cast<scalar_t>(sqrt(664. + sqrt(438976.)) - sqrt(304.) - 19.);
            return 2;
        case 5:
            poles[0] = static_cast<scalar_t>(sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5);
            poles[1] = static_cast<scalar_t>(sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5);
            return 2;
        case 6:
            poles[0] = static_cast<scalar_t>(-0.48829458930304475513011803888378906211227916123937760839);
            poles[1] = static_cast<scalar_t>(-0.081679271076237512597937765737059080653379610398148178525368);
            poles[2] = static_cast<scalar_t>(-0.00141415180832581775108724397655859252786416905534669851652709);
            return 3;
        case 7:
            poles[0] = static_cast<scalar_t>(-0.5352804307964381655424037816816460718339231523426924148812);
            poles[1] = static_cast<scalar_t>(-0.122554615192326690515272264359357343605486549427295558490763);
            poles[2] = static_cast<scalar_t>(-0.0091486948096082769285930216516478534156925639545994482648003);
            return 3;
    }
    return -1;
}

template <typename scalar_t>
inline __device__ scalar_t get_gain(scalar_t * poles, int npoles)
{
    double lam = 1.;
    for (int i=0; i<npoles; ++i, ++poles) {
        double pole = static_cast<double>(*poles);
        lam *= (1. - pole) * (1. - 1./pole);
    }
    return static_cast<scalar_t>(lam);
}

template <bound::type B> struct utils {
    // ZERO & DCT1
    template <typename scalar_t, typename reduce_t, typename offset_t>
    static inline __device__ scalar_t initial(scalar_t * inp, reduce_t pole,
                                              offset_t size, offset_t stride)
    { return _splinc::dct1_initial(inp, pole, size, stride); }
    template <typename scalar_t, typename reduce_t, typename offset_t>
    static inline __device__ scalar_t final(scalar_t * inp, reduce_t pole,
                                            offset_t size, offset_t stride)
    { return _splinc::dct1_final(inp, pole, size, stride); }
};

template <> struct utils<bound::type::Replicate> {
    template <typename scalar_t, typename reduce_t, typename offset_t>
    static inline __device__ scalar_t initial(scalar_t * inp, reduce_t pole,
                                              offset_t size, offset_t stride)
    { return _splinc::dct2_initial(inp, pole, size, stride); }
    template <typename scalar_t, typename reduce_t, typename offset_t>
    static inline __device__ scalar_t final(scalar_t * inp, reduce_t pole,
                                            offset_t size, offset_t stride)
    { return _splinc::dct2_final(inp, pole, size, stride); }
};

template <> struct utils<bound::type::DCT2> {
    template <typename scalar_t, typename reduce_t, typename offset_t>
    static inline __device__ scalar_t initial(scalar_t * inp, reduce_t pole,
                                              offset_t size, offset_t stride)
    { return _splinc::dct2_initial(inp, pole, size, stride); }
    template <typename scalar_t, typename reduce_t, typename offset_t>
    static inline __device__ scalar_t final(scalar_t * inp, reduce_t pole,
                                            offset_t size, offset_t stride)
    { return _splinc::dct2_final(inp, pole, size, stride); }
};

template <> struct utils<bound::type::DFT> {
    template <typename scalar_t, typename reduce_t, typename offset_t>
    static inline __device__ scalar_t initial(scalar_t * inp, reduce_t pole,
                                              offset_t size, offset_t stride)
    { return _splinc::dft_initial(inp, pole, size, stride); }
    template <typename scalar_t, typename reduce_t, typename offset_t>
    static inline __device__ scalar_t final(scalar_t * inp, reduce_t pole,
                                            offset_t size, offset_t stride)
    { return _splinc::dft_final(inp, pole, size, stride); }
};

template <bound::type B, typename scalar_t, typename reduce_t, typename offset_t>
inline __device__ void filter(scalar_t * inp, offset_t size, offset_t stride,
                              const reduce_t * poles, int npoles)
{
    using bound_utils = utils<B>;

    if (size == 1) return;

    reduce_t lam = get_gain(poles, npoles);
    for (offset_t i = 0; i < size; ++i)
        inp[i * stride] *= lam;

    scalar_t *prev, *cur;
    for (int k = 0; k < npoles; ++k) {
        reduce_t pole = poles[k];
        cur = inp;

        *cur = bound_utils::initial(inp, pole, size, stride);

        prev = cur;
        cur = prev + stride;
        for (offset_t i = 1; i < size; ++i, cur += stride) {
            *cur += pole * (*prev);
            prev = cur;
        }

        cur -= stride;
        *cur = bound_utils::final(inp, pole, size, stride);

        prev = cur;
        cur = prev - stride;
        for (offset_t i = size-2; i >= 0; --i, cur -= stride) {
            *cur = pole * ((*prev) - (*cur));
            prev = cur;
        }
    }
}

} // namespace splinc
} // namespace jf

#endif // JF_SPLINC
