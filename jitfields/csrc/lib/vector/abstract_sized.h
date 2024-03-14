#ifndef JF_VECTOR_ABSTRACTSIZED_H
#define JF_VECTOR_ABSTRACTSIZED_H
#include <stdexcept>
#include "forward.h"
#include "abstract_ptr.h"
#include "../cuda_switch.h"

// Abstract base class for "sized" pointers.
//
// Such objects further define the method `size()`, as well as `end()`
// and `rbegin()` iterators.

namespace jf {

namespace internal {
namespace abstractsized {

// =====================================================================
// 1. Implement reverse iterators and safe accessors
// =====================================================================

template <typename T, unsigned long N=DynamicSize, long S=1, typename D=void>
class Base: public AbstractPointer<T, S, Base<T,N,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = Base<T,N,S,D>;
    using this_type_final = Base<T,N,S>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

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
    virtual ~Base() {}

    //--- constructor --------------------------------------------------

    using AbstractPointer<T, S, Base<T,N,S,D> >::AbstractPointer;

    //--- virtual ----------------------------------------------------

    __host__ __device__
    virtual size_type size() const = 0;

    //--- accessors ----------------------------------------------------

    __host__ __device__
    inline reference at(offset_type i)
    {
        if (i < 0 || i >= this->size())
            throw std::invalid_argument("Attempting to access out-of-bound element");
        return (*this)[i];
    }

    __host__ __device__
    inline const_reference at(offset_type i) const
    {
        if (i < 0 || i >= this->size())
            throw std::invalid_argument("Attempting to access out-of-bound element");
        return (*this)[i];
    }

    //--- iterators ----------------------------------------------------

    __host__ __device__
    inline iterator end() const
    {
        return iterator(
            this->data() + this->size() * this->stride(), this->stride());
    }

    __host__ __device__
    inline reverse_iterator rbegin() const
    {
        return reverse_iterator(
            this->data() + (this->size() - 1) * this->stride(),
            -this->stride());
    }

    __host__ __device__
    inline const_iterator cend() const
    {
        return static_cast<const_iterator>(this->end());
    }

    __host__ __device__
    inline const_reverse_iterator crbegin() const
    {
        return static_cast<const_reverse_iterator>(this->rbegin());
    }
};

// =====================================================================
// 2. Static vs Dynamic size
// =====================================================================

template <typename T, unsigned long N=DynamicSize, long S=1, typename D=void>
class Impl:  public Base<T, N, S, Impl<T,N,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = Impl<T,N,S,D>;
    using this_type_final = Impl<T,N,S>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = Base<T, N, S, this_type>;
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
    virtual ~Impl() {}

    //--- constructor --------------------------------------------------

    using Base<T, N, S, Impl<T,N,S,D> >::Base;

    __host__ __device__
    Impl(size_type size = N, offset_type stride =  S):
        parent_type::Base(stride)
    {
        if (N != size) throw std::runtime_error("size not consistent");
    }

    //--- make concrete ------------------------------------------------

    __host__ __device__
    inline size_type size() const { return N; }
};


template <typename T, long S, typename D>
class Impl<T, DynamicSize, S, D>: public Base<T, DynamicSize, S, Impl<T,DynamicSize,S,D> >
{
public:

    //--- types --------------------------------------------------------

    using this_type = Impl<T,DynamicSize,S,D>;
    using this_type_final = Impl<T,DynamicSize,S>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = Base<T, DynamicSize, S, this_type>;
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
    virtual ~Impl() {}

    //--- constructor --------------------------------------------------

    using Base<T, DynamicSize, S, Impl<T,DynamicSize,S,D> >::Base;

    __host__ __device__
    Impl(size_type size = 0, offset_type stride =  S):
        parent_type::Base(stride), _size(size)
    {}

    //--- make concrete ------------------------------------------------

    __host__ __device__
    inline size_type size() const { return this->_size; }

protected:
    size_type _size;
};

} // namespace abstractsized
} // namespace internal

// =====================================================================
// Final. Public class
// =====================================================================

template <typename T, unsigned long N, long S, typename D>
class AbstractSizedPointer:
public internal::abstractsized::Impl<T, N, S, AbstractSizedPointer<T,N,S,D> >
{
public:

    //--- types --------------------------------------------------------

    using this_type = AbstractSizedPointer<T,N,S,D>;
    using this_type_final = AbstractSizedPointer<T,N,S>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = internal::abstractsized::Impl<T, N, S, this_type>;
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
    virtual ~AbstractSizedPointer() {}

    //--- constructor --------------------------------------------------

    using internal::abstractsized::Impl<T, N, S, AbstractSizedPointer<T,N,S,D> >::Impl;

};

template <typename T, long S=1, typename D=void>
using AbstractDynamicSizedPointer = AbstractSizedPointer<T, DynamicSize, S, D>;


} // namespace jf

#endif // JF_VECTOR_ABSTRACTSIZED_H
