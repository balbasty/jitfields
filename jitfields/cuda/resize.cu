const interpolation::type Z = interpolation::type::Nearest;
const interpolation::type L = interpolation::type::Linear;
const interpolation::type Q = interpolation::type::Quadratic;
const interpolation::type C = interpolation::type::Cubic;
const int one = 1;

template <int D,
          interpolation::type IX,    bound::type BX,
          interpolation::type IY=IX, bound::type BY=BX,
          interpolation::type IZ=IY, bound::type BZ=BY>
struct Multiscale {};


template <bound::type B> struct Multiscale<one, Z, B, Z, B, Z, B> {
    using bound_utils = bound::utils<B>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, // loc/size/stride
                reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix = static_cast<offset_t>(floor(x+0.5));
        signed char sx = bound_utils::sign( ix, nw);
        ix = bound_utils::index(ix, nw) * sw;
        *out = bound::get(inp, ix, sx);
    }
};

template <bound::type B> struct Multiscale<one, L, B, L, B, L, B> {
    using bound_utils = bound::utils<B>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, // loc/size/stride
                reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix0 = static_cast<offset_t>(floor(x));
        offset_t ix1 = ix0 + 1;
        reduce_t dx1 = x - ix0;
        reduce_t dx0 = 1 - dx1;
        signed char  sx0 = bound_utils::sign(ix0, nw);
        signed char  sx1 = bound_utils::sign(ix1, nw);
        ix0 = bound_utils::index(ix0, nw) * sw;
        ix1 = bound_utils::index(ix1, nw) * sw;

        *out = static_cast<scalar_t>(
                  static_cast<reduce_t>(bound::get(inp, ix0, sx0)) * dx0
                + static_cast<reduce_t>(bound::get(inp, ix1, sx1)) * dx1);
    }
};

template <bound::type B> struct Multiscale<one, Q, B, Q, B, Q, B> {
    using bound_utils = bound::utils<B>;
    using inter_utils = interpolation::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, // loc/size/stride
                reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x+0.5));
        reduce_t dx1 = inter_utils::weight(x - ix1);
        reduce_t dx0 = inter_utils::fastweight(x - (ix1 - 1));
        reduce_t dx2 = inter_utils::fastweight((ix1 + 1) - x);
        signed char  sx0 = bound_utils::sign(ix1-1, nw);
        signed char  sx2 = bound_utils::sign(ix1+1, nw);
        signed char  sx1 = bound_utils::sign(ix1,   nw);
        offset_t ix0, ix2;
        ix0 = bound_utils::index(ix1-1, nw) * sw;
        ix2 = bound_utils::index(ix1+1, nw) * sw;
        ix1 = bound_utils::index(ix1,   nw) * sw;

        *out = static_cast<scalar_t>(
                  static_cast<reduce_t>(bound::get(inp, ix0, sx0)) * dx0
                + static_cast<reduce_t>(bound::get(inp, ix1, sx1)) * dx1
                + static_cast<reduce_t>(bound::get(inp, ix2, sx2)) * dx2);
    }
};


template <bound::type B> struct Multiscale<one, C, B, C, B, C, B> {
    using bound_utils = bound::utils<B>;
    using inter_utils = interpolation::utils<C>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, // loc/size/stride
                reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        reduce_t dx1 = inter_utils::fastweight(x - ix1);
        reduce_t dx0 = inter_utils::fastweight(x - (ix1 - 1));
        reduce_t dx2 = inter_utils::fastweight((ix1 + 1) - x);
        reduce_t dx3 = inter_utils::fastweight((ix1 + 2) - x);
        signed char  sx0 = bound_utils::sign(ix1-1, nw);
        signed char  sx2 = bound_utils::sign(ix1+1, nw);
        signed char  sx3 = bound_utils::sign(ix1+2, nw);
        signed char  sx1 = bound_utils::sign(ix1,   nw);
        offset_t ix0, ix2, ix3;
        ix0 = bound_utils::index(ix1-1, nw) * sw;
        ix2 = bound_utils::index(ix1+1, nw) * sw;
        ix3 = bound_utils::index(ix1+2, nw) * sw;
        ix1 = bound_utils::index(ix1,   nw) * sw;

        *out = static_cast<scalar_t>(
                  static_cast<reduce_t>(bound::get(inp, ix0, sx0)) * dx0
                + static_cast<reduce_t>(bound::get(inp, ix1, sx1)) * dx1
                + static_cast<reduce_t>(bound::get(inp, ix2, sx2)) * dx2
                + static_cast<reduce_t>(bound::get(inp, ix3, sx3)) * dx3);
    }
};

template <interpolation::type IX, bound::type BX,
          typename scalar_t, typename offset_t>
__global__ void kernel1d(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x = 7;
        offset_t batch_offset = index2offset_1d(i, ndim, size_out, stride_inp, x);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<one, IX, BX>::resize(out + out_offset, inp + batch_offset,
                                        x, size_inp[ndim-1], stride_inp[ndim-1],
                                        scale[ndim-1], shift);
    }
}