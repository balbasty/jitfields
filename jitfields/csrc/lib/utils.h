#ifndef JF_UTILS
#define JF_UTILS
#include "cuda_switch.h"

namespace jf {

// static check for floating types
template <typename T>
struct is_floating_point { static constexpr bool value = false; };
template <>
struct is_floating_point<float> { static constexpr bool value = true; };
template <>
struct is_floating_point<double> { static constexpr bool value = true; };
#ifdef __CUDACC__
template <>
struct is_floating_point<half> { static constexpr bool value = true; };
#endif


template <typename T>
inline __device__
void swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}

template <typename T>
inline __device__
T square(T a)
{
    return a*a;
}

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
T pow(T a, int N) {
    T p = a;
#   pragma unroll
    for(int d = 0; d < N-1; ++d)
        p *= a;
    return p;
}

template <typename T>
inline __device__
T min(T a, T b)
{
    return (a < b ? a : b);
}

template <typename T>
inline __device__
T max(T a, T b)
{
    return (a > b ? a : b);
}

template <typename T>
inline __device__
T abs(T a)
{
    return static_cast<T>(a < 0 ? -a : a);
}

template <typename T>
inline __device__
signed char sign(T a)
{
    return static_cast<signed char>(a == 0 ? 0 : a < 0 ? -1 : 1);
}

#ifdef __CUDACC__
template <>
inline __device__
half min<>(half a, half b)
{
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return (a < b ? a : b);
}
template <>
inline __device__
half max<>(half a, half b)
{
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return (a > b ? a : b);
}
#endif

// fmod
template <typename T, typename U,
          bool is_float_T = is_floating_point<T>::value,
          bool is_float_U = is_floating_point<U>::value >
struct _mod
{
    inline __device__ static
    T f(T x, U d)
    {
        signed char sx = sign(x);
        signed char sd = sign(d);

        long ratio = (sx*sd)*static_cast<long>(trunc(abs(x)/abs(d)));
        return (x - ratio * d);
    }
};


template <typename T, typename U>
struct _mod<T, U, false, false>
{
    inline __device__ static
    T f(T x, U d)
    {
        return x % d;
    }
};

template <typename T, typename U>
inline __device__
T mod(T x, U d)
{
    return _mod<T,U>::f(x, d);
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
inline __device__
void fillfrom(U out[N], const V * inp)
{
#   pragma unroll
    for (int n=0; n < N; ++ n)
        out[n] = static_cast<U>(inp[n]);
}

template <int N, typename U, typename V, typename W>
inline __device__
void fillfrom(U out[N], const V * inp, W stride)
{
#   pragma unroll
    for (int n=0; n < N; ++n, inp += stride)
        out[n] = static_cast<U>(*inp);
}


template <int N, typename U, typename V>
inline __device__
void fill(U * out, V inp)
{
    auto val = static_cast<U>(inp);
#   pragma unroll
    for (int n=0; n < N; ++n)
        out[n] = val;
}

template <int N, typename U, typename V, typename W>
inline __device__
void fill(U * out, V inp, W stride)
{
    auto val = static_cast<U>(inp);
#   pragma unroll
    for (int n=0; n < N; ++n, out += stride)
        (*out) = val;
}

} // namespace jf

#endif // JF_UTILS
