#ifndef JF_UTILS
#define JF_UTILS
#include "cuda_switch.h"

namespace jf {

template <typename T>
inline __device__
T square(T a) { return a*a; }

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

} // namespace jf

#endif // JF_UTILS
