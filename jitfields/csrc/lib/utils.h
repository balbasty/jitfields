#ifndef JF_UTILS
#define JF_UTILS
#include "cuda_switch.h"

namespace jf {

template <typename T>
inline __device__
void swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}

template <typename T>
inline __device__
T square(T a) { return a*a; }

template <int N, typename T>
inline __device__
T pow(T a) {
    T p = a;
#   pragma unroll
    for(int d = 0; d < N-1; ++d)
        p *= a;
    return p;
}

template <typename T>
inline __device__
T min(T a, T b) { return (a < b ? a : b); }

template <typename T>
inline __device__
T max(T a, T b) { return (a > b ? a : b); }

#ifdef __CUDACC__
template <>
inline __device__
half min<>(half a, half b) {
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return (a < b ? a : b);
}
template <>
inline __device__
half max<>(half a, half b) {
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return (a > b ? a : b);
}
#endif

template <typename T>
inline __device__
T remainder(T x, T d)
{
    return (x - (x / d) * d);
}

template <typename T, typename size_t>
inline __device__
T prod(const T * x, size_t size)
{
    if (size == 0)
        return static_cast<T>(1);
    T tmp = x[0];
    for (size_t d = 1; d < size; ++d)
        tmp *= x[d];
    return tmp;
}

template <size_t size, typename T>
inline __device__
T prod(const T * x)
{
    if (size == 0)
        return static_cast<T>(1);
    T tmp = x[0];
#   pragma unroll
    for (size_t d = 1; d < size; ++d)
        tmp *= x[d];
    return tmp;
}

template <int N, typename U, typename V>
void fillfrom(U out[N], const V * inp)
{
#   pragma unroll
    for (int n=0; n < N; ++ n)
        out[n] = static_cast<U>(inp[n]);
}

template <int N, typename U, typename V, typename W>
void fillfrom(U out[N], const V * inp, W stride)
{
#   pragma unroll
    for (int n=0; n < N; ++n, inp += stride)
        out[n] = static_cast<U>(*inp);
}


template <int N, typename U, typename V>
void fill(U * out, V inp)
{
    auto val = static_cast<U>(inp);
#   pragma unroll
    for (int n=0; n < N; ++n)
        out[n] = val;
}

template <int N, typename U, typename V, typename W>
void fill(U * out, V inp, W stride)
{
    auto val = static_cast<U>(inp);
#   pragma unroll
    for (int n=0; n < N; ++n, out += stride)
        (*out) = val;
}

} // namespace jf

#endif // JF_UTILS
