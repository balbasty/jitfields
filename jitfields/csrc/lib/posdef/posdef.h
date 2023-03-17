#ifndef JF_POSDEF
#define JF_POSDEF
#include "../cuda_switch.h"
#include "../utils.h"
#include "utils.h"
#include "cholesky.h"

#define JFH_OnePlusTiny 1.000001

namespace jf {
namespace posdef {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                Enum
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum class type: unsigned char {
    None,         // No Hessian provided so nothing to do
    Eye,          // Scaled identity
    Diag,         // Diagonal matrix
    ESTATICS,     // (C-1) elements are independent conditioned on the last one
    Sym,          // Compact symmetric matrix
    Full          // Full matrix (but must be symmetric)
};

template <typename offset_t>
static inline
type guess_type(offset_t C, offset_t CC)
{
    if (CC == 0)
        return type::None;
    else if (CC == 1)
        return type::Eye;
    else if (CC == C)
        return type::Diag;
    else if (CC == 2*C-1)
        return type::ESTATICS;
    else if (CC == (C*(C+1))/2)
        return type::Sym;
    else if (CC == C*C)
        return type::Full;
    else
        throw std::runtime_error("Input does not look like a known matrix form");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            Static traits
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Function common to all Hessian types
///
/// All functions are templated and work with any scalar / pointer types.
/// We list them here with int/float for readability.
/// Functions can take an additional unused argument (not listed here)
/// whose data type is used for intermediate computation, if provided.
///
/// matvec(int C, float * x, const float * v, const float * h, const float * w = nullptr)
/// >>    x = (H + diag(w)) * v
/// addmatvec_(int C, float * x, const float * v, const float * h, const float * w = nullptr)
/// >>    x += (H + diag(w)) * v
/// submatvec_(int C, float * x, const float * v, const float * h, const float * w = nullptr)
/// >>    x -= (H + diag(w)) * v
///
/// invert(int C, float * a, const float * h, const float * w = nullptr)
/// >>    A = inv(H + diag(w))
/// solve_(int C, float * v, const float * h, const float * w = nullptr)
/// >>    v = (H + diag(w)) \ v
/// solve(int C, float * x, const float * v, const float * h, const float * w = nullptr)
/// >>    x = (H + diag(w)) \ v
/// relax_(int C, float * x, float * v, const float * h, const float * w = nullptr)
/// >>    x += (H + diag(w)) \ (v - H * x)
template <class Child, typename offset_t=long, int C=0>
struct common
{
    using this_type = common<Child, offset_t, C>;

    static constexpr bool need_buffer = false;
    static constexpr offset_t work_size = 0;

    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void solve_(
        vptr_t v,
        hptr_t h,
        wptr_t w = nullptr,
        bptr_t b = nullptr,
        reduce_t unused = static_cast<reduce_t>(0))
    {
        Child::solve_impl_(v, h, w, b, unused);  // v <- (H + diag(w)) \ v
    }

    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<xptr_t, vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void solve(
        xptr_t x,
        vptr_t v,
        hptr_t h,
        wptr_t w = nullptr,
        bptr_t b = nullptr,
        reduce_t unused = static_cast<reduce_t>(0))
    {
        copy_(x, v);                             // x <- v
        Child::solve_impl_(x, h, w, b, unused);  // x <- (H + diag(w)) \ x
    }

    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<xptr_t, vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void relax_(
        xptr_t x,
        hptr_t h,
        vptr_t v,
        wptr_t w = nullptr,
        bptr_t b = nullptr,
        reduce_t unused = static_cast<reduce_t>(0))
    {
        Child::submatvec_(v, h, x, unused);          // v <- v - H * x
        Child::solve_impl_(v, h, w, b, unused);      // v <- (H + diag(w)) \ v
        add_(x, v, unused);                          // x <- x + v
    }

    // The next few functions are aliases that do not expect a buffer

    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void solve_(
        vptr_t v,
        hptr_t h,
        wptr_t w,
        double unused)
    {
        const void * b = nullptr;
        return this_type::solve_(v, h, w, b, unused);
    }

    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void solve_(
        vptr_t v,
        hptr_t h,
        wptr_t w,
        float unused)
    {
        const void * b = nullptr;
        return this_type::solve_(v, h, w, b, unused);
    }

#ifdef __CUDACC__
    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void solve_(
        vptr_t v,
        hptr_t h,
        wptr_t w,
        half unused)
    {
        const void * b = nullptr;
        return this_type::solve_(v, h, w, b, unused);
    }
#endif

    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void solve(
        xptr_t x,
        vptr_t v,
        hptr_t h,
        wptr_t w,
        double unused)
    {
        const void * b = nullptr;
        return this_type::solve(x, v, h, w, b, unused);
    }

    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void solve(
        xptr_t x,
        vptr_t v,
        hptr_t h,
        wptr_t w,
        float unused)
    {
        const void * b = nullptr;
        return this_type::solve(x, v, h, w, b, unused);
    }

#ifdef __CUDACC__
    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void solve_(
        xptr_t x,
        vptr_t v,
        hptr_t h,
        wptr_t w,
        half unused)
    {
        const void * b = nullptr;
        return this_type::solve(x, v, h, w, b, unused);
    }
#endif

    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void relax_(
        xptr_t x,
        hptr_t h,
        vptr_t v,
        wptr_t w,
        float unused)
    {
        const void * b = nullptr;
        return this_type::relax_(x, h, v, w, b, unused);
    }

    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void relax_(
        xptr_t x,
        hptr_t h,
        vptr_t v,
        wptr_t w,
        double unused)
    {
        const void * b = nullptr;
        return this_type::relax_(x, h, v, w, b, unused);
    }

#ifdef __CUDACC__
    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *
        >
    static inline __device__
    void relax_(
        xptr_t x,
        hptr_t h,
        vptr_t v,
        wptr_t w,
        half unused)
    {
        const void * b = nullptr;
        return this_type::relax_(x, h, v, w, b, unused);
    }
#endif

    template <typename optr_t, typename iptr_t>
    static inline __device__
    void copy_(optr_t out, iptr_t inp)
    {
        using output_t = typename internal::elem_type<optr_t>::value;
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            out[c] = static_cast<output_t>(inp[c]);
    }

    template <typename optr_t, typename iptr_t>
    static inline __device__
    void copy_(offset_t L, optr_t out, iptr_t inp)
    {
        using output_t = typename internal::elem_type<optr_t>::value;
        for (offset_t c = 0; c < L; ++c)
            out[c] = static_cast<output_t>(inp[c]);
    }

    template <typename optr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<optr_t, iptr_t>::value>
    static inline __device__
    void add_(optr_t out, iptr_t inp,
              reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            internal::iadd<reduce_t>(out[c], inp[c]);
    }
};


// Dynamic-sized specialization
template <class Child, typename offset_t>
struct common<Child, offset_t, 0>
{
    using this_type = common<Child, offset_t, 0>;

    static constexpr bool need_buffer = false;

    /// Buffer size (in number of elements) per voxels.
    /// Most Hessian types can be used as is and no not need a buffer.
    /// Only full and symmetric matrices need a buffer (for Cholesky)
    static inline offset_t work_size(offset_t C)
    {
        return static_cast<offset_t>(0);
    }

    /// Solve the linear system in-place: v = (H + diag(w)) \ v
    ///
    /// @param C            number of channels
    /// @param v[inout]     pointer to value at which to solve (length C)
    /// @param h[in]        pointer to hessian (size CxC, may be compact)
    /// @param w[in]        pointer to regulariser weights (length C)
    /// @param b[inout]     pointer to temporary buffer (length `worksize`)
    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void solve_(
        offset_t C,
        vptr_t v,
        hptr_t h,
        wptr_t w = nullptr,
        bptr_t b = nullptr,
        reduce_t unused = static_cast<reduce_t>(0))
    {
        Child::solve_impl_(C, v, h, w, b, unused);  // v <- (H + diag(w)) \ v
    }

    /// Solve the linear system: x = (H + diag(w)) \ v
    ///
    /// @param C      number of channels
    /// @param x      pointer to output array (length C)
    /// @param v      pointer to value at which to solve (length C)
    /// @param h      pointer to hessian (size CxC, may be compact)
    /// @param w      pointer to regulariser weights (length C)
    /// @param b      pointer to temporary buffer (length `worksize`)
    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<xptr_t, vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void solve(
        offset_t C,
        xptr_t x,
        vptr_t v,
        hptr_t h,
        wptr_t w = nullptr,
        bptr_t b = nullptr,
        reduce_t unused = static_cast<reduce_t>(0))
    {
        copy_(C, x, v);                             // x <- v
        Child::solve_impl_(C, x, h, w, b, unused);  // x <- (H + diag(w)) \ x
    }

    /// Solve the linear system and increment x += (H + diag(w)) \ (v - H * x)
    /// where v contains `g - W * x`
    ///
    /// Let H be a block-diagonal matrix and W is a large matrix.
    /// The linear system (H + W) * x = g can be solved by iterative
    /// relaxation (e.g. Jacobi/Gauss-Seidel). Each iteration is:
    ///   x += (H + diag(diag(W))) \ (g - (H + W) * x)
    /// which can be seen as one iteration of preconditioned gradient descent,
    /// where `H + diag(diag(W))` (= the block diagonal of `H + W`) is
    /// the preconditioner.
    ///
    /// @param C      number of channels
    /// @param x      pointer to output array (length C)
    /// @param v      pointer to value at which to solve (length C)
    /// @param h      pointer to hessian (size CxC, may be compact)
    /// @param w      pointer to regulariser weights (length C)
    /// @param b      pointer to temporary buffer (length `worksize`)
    template <
        typename xptr_t,
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<xptr_t, vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void relax_(
        offset_t C,
        xptr_t x,
        hptr_t h,
        vptr_t v,
        wptr_t w = nullptr,
        bptr_t b = nullptr,
        reduce_t unused = static_cast<reduce_t>(0))
    {
        Child::submatvec_(C, v, h, x, unused);          // v <- v - H * x
        Child::solve_impl_(C, v, h, w, b, unused);      // v <- (H + diag(w)) \ v
        add_(C, x, v, unused);                          // x <- x + v
    }

    // Redirect solve_(C, v, h, w, unused) to solve_(C, v, h, w, nullptr, unused)
    // when unused is a scalar type (and not a pointer).
    // Because of the use of templates, I have to write it down for each
    // scalar type (maybe there's a more programmatic way to do it with
    // traits and SFINAE?).

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *>
    static inline __device__
    void solve_(offset_t C, vptr_t v, hptr_t h, wptr_t w, double unused)
    {
        const void * b = nullptr;
        return this_type::solve_(C, v, h, w, b, unused);
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *>
    static inline __device__
    void solve_(offset_t C, vptr_t v, hptr_t h, wptr_t w, float unused)
    {
        const void * b = nullptr;
        return this_type::solve_(C, v, h, w, b, unused);
    }

#ifdef __CUDACC__
    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *>
    static inline __device__
    void solve_(offset_t C, vptr_t v, hptr_t h, wptr_t w, half unused)
    {
        const void * b = nullptr;
        return this_type::solve_(C, v, h, w, b, unused);
    }
#endif

    template <typename xptr_t, typename vptr_t,
              typename hptr_t, typename wptr_t = const void *>
    static inline __device__
    void solve(offset_t C, xptr_t x, vptr_t v, hptr_t h, wptr_t w, double unused)
    {
        const void * b = nullptr;
        return this_type::solve(C, x, v, h, w, b, unused);
    }

    template <typename xptr_t, typename vptr_t,
              typename hptr_t, typename wptr_t = const void *>
    static inline __device__
    void solve(offset_t C, xptr_t x, vptr_t v, hptr_t h, wptr_t w, float unused)
    {
        const void * b = nullptr;
        return this_type::solve(C, x, v, h, w, b, unused);
    }

#ifdef __CUDACC__
    template <typename xptr_t, typename vptr_t,
              typename hptr_t, typename wptr_t = const void *>
    static inline __device__
    void solve_(offset_t C, xptr_t x, vptr_t v, hptr_t h, wptr_t w, half unused)
    {
        const void * b = nullptr;
        return this_type::solve(C, x, v, h, w, b, unused);
    }
#endif

    template <typename xptr_t, typename vptr_t,
              typename hptr_t, typename wptr_t = const void *>
    static inline __device__
    void relax_(offset_t C, xptr_t x, hptr_t h, vptr_t v, wptr_t w, float unused)
    {
        const void * b = nullptr;
        return this_type::relax_(C, x, v, h, w, b, unused);
    }

    template <typename xptr_t, typename vptr_t,
              typename hptr_t, typename wptr_t = const void *>
    static inline __device__
    void relax_(offset_t C, xptr_t x, hptr_t h, vptr_t v, wptr_t w, double unused)
    {
        const void * b = nullptr;
        return this_type::relax_(C, x, v, h, w, b, unused);
    }

#ifdef __CUDACC__
    template <typename xptr_t, typename vptr_t,
              typename hptr_t, typename wptr_t = const void *>
    static inline __device__
    void relax_(offset_t C, xptr_t x, hptr_t h, vptr_t v, wptr_t w, half unused)
    {
        const void * b = nullptr;
        return this_type::relax_(C, x, v, h, w, b, unused);
    }
#endif

//protected:

    /// Copy values into a strided vector
    ///
    /// @param C          vector length
    /// @param out[out]   pointer to output vector (length C)
    /// @param inp[in]    pointer to input vector (length C)
    template <typename optr_t, typename iptr_t>
    static inline __device__
    void copy_(offset_t C, optr_t out, iptr_t inp)
    {
        using output_t = typename internal::elem_type<optr_t>::value;
        for (offset_t c = 0; c < C; ++c)
            out[c] = static_cast<output_t>(inp[c]);
    }

    /// Add values into a strided vector
    ///
    /// @param C             vector length
    /// @param out[inout]    pointer to output vector (length C)
    /// @param inp[in]       pointer to input vector (length C)
    template <typename optr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<optr_t, iptr_t>::value>
    static inline __device__
    void add_(offset_t C, optr_t out, iptr_t inp,
              reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c)
            internal::iadd<reduce_t>(out[c], inp[c]);
    }
};

template <type hessian_t, typename offset_t=long, int C=0>
struct utils: public common<utils<hessian_t, offset_t, C>, offset_t, C>
{
    /// Return the matrix-vector product to a vector: out = H * inp
    ///
    /// @param o[out]    Output vector (length C)
    /// @param i[in]     Input vector (length C)
    /// @param h[in]     Matrix (size CxC, may be compact)
    template <typename iptr_t, typename hptr_t, typename optr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, iptr_t, hptr_t>::value>
    static inline __device__ void
    matvec(optr_t o, hptr_t h, iptr_t i,
           reduce_t unused = static_cast<reduce_t>(0));

    /// Add the matrix-vector product to a vector: out += H * inp
    ///
    /// @param o[inout]  Output vector (length C)
    /// @param h[in]     Matrix (size CxC, may be compact)
    /// @param i[in]     Input vector (length C)
    template <typename iptr_t, typename hptr_t, typename optr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, iptr_t, hptr_t>::value>
    static inline __device__ void
    addmatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t unused = static_cast<reduce_t>(0));

    /// Subtract the matrix-vector product to a vector: out -= H * inp
    ///
    /// @param o[inout]  Output vector (length C)
    /// @param h[in]     Matrix (size CxC, may be compact)
    /// @param i[in]     Input vector (length C)
    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t unused = static_cast<reduce_t>(0));

//protected:

    /// Solve the linear system in-place: v = (H + diag(w)) \ v
    ///
    /// @param C
    /// @param v[inout]    Input/output vector
    /// @param h[in]       Matrix
    /// @param w[in]       Diagonal of the regularizer
    template <typename hptr_t, typename vptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  hptr_t, vptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t b = nullptr,
                reduce_t unused = static_cast<reduce_t>(0));
};


// dynamic-sized specialization
template <type hessian_t, typename offset_t>
struct utils<hessian_t, offset_t, 0>: public common<utils<hessian_t, offset_t, 0>, offset_t, 0>
{
    /// Return the matrix-vector product to a vector: out = H * inp
    ///
    /// @param C         Number of channels
    /// @param o[out]    Output vector (length C)
    /// @param i[in]     Input vector (length C)
    /// @param h[in]     Matrix (size CxC, may be compact)
    template <typename iptr_t, typename hptr_t, typename optr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, iptr_t, hptr_t>::value>
    static inline __device__ void
    matvec(offset_t C, optr_t o, hptr_t h, iptr_t i,
           reduce_t unused = static_cast<reduce_t>(0));

    /// Add the matrix-vector product to a vector: out += H * inp
    ///
    /// @param C         Number of channels
    /// @param o[inout]  Output vector (length C)
    /// @param h[in]     Matrix (size CxC, may be compact)
    /// @param i[in]     Input vector (length C)
    template <typename iptr_t, typename hptr_t, typename optr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, iptr_t, hptr_t>::value>
    static inline __device__ void
    addmatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t unused = static_cast<reduce_t>(0));

    /// Subtract the matrix-vector product to a vector: out -= H * inp
    ///
    /// @param C         Number of channels
    /// @param o[inout]  Output vector (length C)
    /// @param h[in]     Matrix (size CxC, may be compact)
    /// @param i[in]     Input vector (length C)
    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t unused = static_cast<reduce_t>(0));

//protected:

    /// Solve the linear system in-place: v = (H + diag(w)) \ v
    ///
    /// @param C
    /// @param v[inout]    Input/output vector
    /// @param h[in]       Matrix
    /// @param w[in]       Diagonal of the regularizer
    template <typename hptr_t, typename vptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  hptr_t, vptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(offset_t C, vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t b = nullptr,
                reduce_t unused = static_cast<reduce_t>(0));
};

// aliases to make the following code less ugly
template <typename offset_t, int C> using utils_none      = utils<type::None, offset_t, C>;
template <typename offset_t, int C> using utils_eye       = utils<type::Eye, offset_t, C>;
template <typename offset_t, int C> using utils_diag      = utils<type::Diag, offset_t, C>;
template <typename offset_t, int C> using utils_estatics  = utils<type::ESTATICS, offset_t, C>;
template <typename offset_t, int C> using utils_sym       = utils<type::Sym, offset_t, C>;
template <typename offset_t, int C> using common_none     = common<utils_none<offset_t, C>, offset_t, C>;
template <typename offset_t, int C> using common_eye      = common<utils_eye<offset_t, C>, offset_t, C>;
template <typename offset_t, int C> using common_diag     = common<utils_diag<offset_t, C>, offset_t, C>;
template <typename offset_t, int C> using common_estatics = common<utils_estatics<offset_t, C>, offset_t, C>;
template <typename offset_t, int C> using common_sym      = common<utils_sym<offset_t, C>, offset_t, C>;


template <typename offset_t, int C>
struct utils<type::None, offset_t, C>: public common_none<offset_t, C>
{
    template <
        typename iptr_t,
        typename hptr_t,
        typename optr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void matvec(
        optr_t /*o*/,
        hptr_t /*h*/,
        iptr_t /*i*/,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {}

    template <
        typename iptr_t,
        typename hptr_t,
        typename optr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void addmatvec_(
        optr_t /*o*/,
        hptr_t /*h*/,
        iptr_t /*i*/,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {}

    template <
        typename iptr_t,
        typename hptr_t,
        typename optr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void submatvec_(
        optr_t /*o*/,
        hptr_t /*h*/,
        iptr_t /*i*/,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {}

    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void solve_impl_(
        vptr_t v,
        hptr_t /*h*/,
        wptr_t w = nullptr,
        bptr_t /*b*/ = nullptr,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        if (w)
#           pragma unroll
            for (offset_t c = 0; c < C; ++c)
                internal::idiv<reduce_t>(v[c], w[c]);
        else
#           pragma unroll
            for (offset_t c = 0; c < C; ++c)
                internal::idiv<reduce_t>(v[c], 0.);
    }
};

template <typename offset_t>
struct utils<type::None, offset_t, 0>: public common_none<offset_t, 0>
{
    template <
        typename iptr_t,
        typename hptr_t,
        typename optr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void matvec(
        offset_t /*C*/,
        optr_t /*o*/,
        hptr_t /*h*/,
        iptr_t /*i*/,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {}

    template <
        typename iptr_t,
        typename hptr_t,
        typename optr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void addmatvec_(
        offset_t /*C*/,
        optr_t /*o*/,
        hptr_t /*h*/,
        iptr_t /*i*/,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {}

    template <
        typename iptr_t,
        typename hptr_t,
        typename optr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void submatvec_(
        offset_t /*C*/,
        optr_t /*o*/,
        hptr_t /*h*/,
        iptr_t /*i*/,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {}

    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void solve_impl_(
        offset_t C,
        vptr_t v,
        hptr_t /*h*/,
        wptr_t w = nullptr,
        bptr_t /*b*/ = nullptr,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        if (w)
            for (offset_t c = 0; c < C; ++c)
                internal::idiv<reduce_t>(v[c], w[c]);
        else
            for (offset_t c = 0; c < C; ++c)
                internal::idiv<reduce_t>(v[c], 0.);
    }
};

#include "eye.inl"
#include "diag.inl"
#include "estatics.inl"
#include "sym.inl"
#include "full.inl"

} // namespace posdef
} // namespace jf

#endif // JF_POSDEF
