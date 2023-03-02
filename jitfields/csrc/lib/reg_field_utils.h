#ifndef JF_REGULARISERS_UTILS
#define JF_REGULARISERS_UTILS
#include "cuda_switch.h"
#include "bounds.h"
#include "utils.h"

namespace jf {
namespace reg_field {

const bound::type B0 = bound::type::NoCheck;
const int zero  = 0;
const int one   = 1;
const int two   = 2;
const int three = 3;

template <int D, bound::type BX=B0, bound::type BY=BX, bound::type BZ=BY>
struct RegField {};

template <int C, int D, bound::type BX=B0, bound::type BY=BX, bound::type BZ=BY>
struct RegFieldStatic {};

//----------------------------------------------------------------------
//          Helpers to implement generic variants that either
//          assign to, add or subtract from the output pointer.
//----------------------------------------------------------------------

template <typename T, typename IT>
inline __device__ T & set(T & out, const IT & in)
{
    out = static_cast<T>(in);
    return out;
}

template <typename T, typename IT>
inline __device__ T & iadd(T & out, const IT & in)
{
    out = static_cast<T>(static_cast<IT>(out) + in);
    return out;
}

template <typename T, typename IT>
inline __device__ T & isub(T & out, const IT & in)
{
    out = static_cast<T>(static_cast<IT>(out) - in);
    return out;
}

template <typename T, typename IT>
inline __device__ T add(const T & out, const IT & in)
{
    return static_cast<T>(static_cast<IT>(out) + in);
}

template <typename T, typename IT>
inline __device__ T sub(const T & out, const IT & in)
{
    return static_cast<T>(static_cast<IT>(out) - in);
}

template <char op, typename scalar_t, typename reduce_t = scalar_t>
struct Op {
    typedef scalar_t & (*FuncType)(scalar_t &, const reduce_t &);
    static constexpr FuncType f = set;
};

template <typename scalar_t, typename reduce_t>
struct Op<'+', scalar_t, reduce_t> {
    typedef scalar_t & (*FuncType)(scalar_t &, const reduce_t &);
    static constexpr FuncType f = iadd;
};

template <typename scalar_t, typename reduce_t>
struct Op<'-', scalar_t, reduce_t> {
    typedef scalar_t & (*FuncType)(scalar_t &, const reduce_t &);
    static constexpr FuncType f = isub;
};

} // namespace reg_field
} // namespace jf

#endif // JF_REGULARISERS_UTILS
