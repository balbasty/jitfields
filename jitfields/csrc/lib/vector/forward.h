#ifndef JF_VECTOR_FORWARD_H
#define JF_VECTOR_FORWARD_H
#include <limits>

namespace jf {

static constexpr unsigned long DynamicSize = std::numeric_limits<unsigned long>::max();
static constexpr long DynamicStride = std::numeric_limits<long>::max();

template <typename T, long S=1, typename D=void>
class AbstractPointer;

template <typename T, long S=1, typename D=void>
class WeakRef;

template <typename T, unsigned long N=DynamicSize, long S=1, typename D=void>
class AbstractSizedPointer;

template <typename T, unsigned long N=DynamicSize, long S=1, typename D=void>
class WeakSizedRef;

template <typename T, unsigned long N=DynamicSize, long S=1, typename D=void>
class AbstractVector;

template <typename T, unsigned long N=DynamicSize, long S=1, typename D=void>
class WeakVector;

template <typename T, unsigned long N=DynamicSize, typename D=void>
class Vector;

}

#endif // JF_VECTOR_FORWARD_H
