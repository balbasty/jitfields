#ifndef JF_VECTOR_WEAKREF_H
#define JF_VECTOR_WEAKREF_H
#include "forward.h"
#include "traits.h"
#include "abstract_ptr.h"
#include "../cuda_switch.h"

namespace jf {

template <typename T, long S, typename D>
class WeakRef: public AbstractPointer<T, S, WeakRef<T,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = typename internal::guess_type<WeakRef<T, S>, D>::value;

    // inherited
    using parent_type = AbstractPointer<T, S, this_type>;
    using value_type = typename parent_type::value_type;
    using size_type = typename parent_type::size_type;
    using offset_type = typename parent_type::offset_type;
    using reference = typename parent_type::reference;
    using const_reference = typename parent_type::const_reference;
    using pointer = typename parent_type::pointer;
    using const_pointer = typename parent_type::const_pointer;
    using iterator = typename parent_type::iterator;
    using const_iterator = typename parent_type::const_iterator;
    using reverse_iterator = typename parent_type::reverse_iterator;
    using const_reverse_iterator = typename parent_type::const_reverse_iterator;
    static constexpr offset_type static_stride = parent_type::static_stride;

    //--- destructor ---------------------------------------------------

    __host__ __device__
    virtual ~WeakRef() {}

    //--- constructors -------------------------------------------------

    __host__ __device__
    WeakRef(offset_type stride):
        parent_type(stride)
    {}

    __host__ __device__
    WeakRef(T * other = nullptr, offset_type stride=S):
        parent_type(stride), _data(other)
    {}

    __host__ __device__
    template <long OS, typename OD>
    WeakRef(const AbstractPointer<T, OS, OD> & other):
        parent_type(other.stride()), _data(other.data())
    {}

    //--- virtual ------------------------------------------------------

    __host__ __device__
    inline T * data() const
    {
        return _data;
    }

protected:
    T * _data;
};


template <typename T, typename D=void>
using DynamicWeakRef = WeakRef<T, DynamicStride, D>;

template <typename T>
__host__ __device__
WeakRef<T> weak_ref(T * ptr)
{
    return WeakRef<T>(ptr);
}

template <typename T>
__host__ __device__
DynamicWeakRef<T> weak_ref(T * ptr, long stride)
{
    return DynamicWeakRef<T>(ptr, stride);
}

template <typename T, long S, typename D>
__host__ __device__
WeakRef<T,S> weak_ref(const AbstractPointer<T,S,D> & ptr)
{
    return WeakRef<T, S>(ptr);
}


} // namespace jf

#endif // JF_VECTOR_WEAKREF_H
