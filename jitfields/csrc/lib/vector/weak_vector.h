#ifndef JF_VECTOR_WEAKVECTOR_H
#define JF_VECTOR_WEAKVECTOR_H
#include "forward.h"
#include "traits.h"
#include "abstract_vector.h"
#include "../cuda_switch.h"

namespace jf {

/***********************************************************************
 *
 *                     STATIC-SIZED VECTOR REF
 *
 **********************************************************************/

template <typename T, unsigned long N, long S, typename D>
class WeakVector: public AbstractVector<T, N, S, WeakVector<T,N,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = WeakVector<T, N, S, D>;
    using this_type_final = WeakVector<T, N, S>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = AbstractVector<T, N, S, this_type>;
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
    virtual ~WeakVector() {}

    //--- constructors -------------------------------------------------

    __host__ __device__
    WeakVector(size_type size, offset_type stride):
        parent_type(size, stride), _data(nullptr)
    {}

    __host__ __device__
    WeakVector(size_type size, T * other = nullptr, offset_type stride = S):
        parent_type(size, stride), _data(other)
    {}

    __host__ __device__
    WeakVector(T * other, offset_type stride = S):
        parent_type(N, stride), _data(other)
    {}

    template <unsigned long ON, long OS, typename OD>
    __host__ __device__
    WeakVector(const AbstractSizedPointer<T, ON, OS, OD> & other):
        parent_type(other.size(), other.stride()), _data(other.data())
    {}

    //--- virtual ------------------------------------------------------

    __host__ __device__
    inline const T * data() const
    {
        return this->_data;
    }

    __host__ __device__
    inline T * data()
    {
        return this->_data;
    }

protected:
    T * _data;
};

template <typename T>
__host__ __device__
WeakVector<T> weak_vec(unsigned long N, T * ptr)
{
    return WeakVector<T>(N, ptr);
}

template <typename T, unsigned long N>
__host__ __device__
WeakVector<T,N> weak_vec(T ptr[N])
{
    return WeakVector<T,N>(N, ptr);
}

template <typename T>
__host__ __device__
WeakVector<T,DynamicSize,DynamicStride> weak_vec(unsigned long N, T * ptr, long stride)
{
    return WeakVector<T,DynamicSize,DynamicStride>(N, ptr, stride);
}

template <typename T, unsigned long N>
__host__ __device__
WeakVector<T,N,DynamicStride> weak_vec(T ptr[N], long stride)
{
    return WeakVector<T,N,DynamicStride>(N, ptr, stride);
}

template <typename T, unsigned N, long S, typename D>
__host__ __device__
WeakVector<T,N,S> weak_vec(const AbstractSizedPointer<T,S,N,D> & ptr)
{
    return WeakVector<T,N,S>(ptr);
}

} // namespace jf

// =====================================================================
//  Unpack API
// =====================================================================

template <typename T, unsigned long N, long S, typename D>
struct std::tuple_size< jf::WeakVector<T, N, S, D> >
{
    using value_type = std::size_t;
    static constexpr value_type value = static_cast<std::size_t>(N);
    using type = std::integral_constant<value_type, value>;
};

template <std::size_t I, typename T, unsigned long N, long S, typename D>
struct std::tuple_element< I, jf::WeakVector<T, N, S, D> >
{
    using type = T;
};

#endif // JF_VECTOR_WEAKVECTOR_H
