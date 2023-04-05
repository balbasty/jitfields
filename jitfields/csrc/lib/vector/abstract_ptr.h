#ifndef JF_VECTOR_ABSTRACTPTR_H
#define JF_VECTOR_ABSTRACTPTR_H
#include <stdexcept>
#include "forward.h"
#include "traits.h"
#include "../cuda_switch.h"

// Abstract base class for objects on which pointer arithmetic can be
// performed.
//
// All such classes define the methods `data()` (which returns a raw
// pointer to memory) and `stride()` (which returns the size in byte
// between two consecutive elements _divided_ by the size of an element).
// They also define classical iterators (although without the `end()`
// variants, since pointers are not sized in general).
//
// Pointer arithmetic takes the stride into consideration. That is,
// `ptr++` moves the underlying raw pointer by `stride()` elements.

namespace jf {

namespace internal {
namespace abstractptr {

// =====================================================================
// 1. Implement accessors and comparisons
// =====================================================================

template <typename T, typename D = void>
class Accessors {
public:

    //--- types --------------------------------------------------------

    using this_type = Accessors<T, D>;
    using this_type_final = Accessors<T>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    using value_type = T;
    using size_type = unsigned long;
    using offset_type = long;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    //--- destructor ---------------------------------------------------

    __host__ __device__
    virtual ~Accessors() {}

    //--- conversion ---------------------------------------------------

    __host__ __device__
    operator bool () const { return this->data() != nullptr; }

    __host__ __device__
    operator T* () const { return this->data(); }

    //--- virtual ------------------------------------------------------

    __host__ __device__
    virtual pointer data() const = 0;

    __host__ __device__
    virtual offset_type stride() const = 0;

    //--- accessors ----------------------------------------------------

    __host__ __device__
    inline reference operator*() const
    {
        return *(this->data());
    }

    __host__ __device__
    inline reference operator[](offset_type i) const
    {
        return this->data()[i*this->stride()];
    }

    //--- comparisons --------------------------------------------------

    __host__ __device__
    inline bool operator== (const T * other) const
    {
        return this->data() == other;
    }

    __host__ __device__
    inline bool operator== (const Accessors<T> & other) const
    {
        return this->data() == other.data();
    }

    __host__ __device__
    inline bool operator!= (const T * other) const
    {
        return this->data() != other;
    }

    __host__ __device__
    inline bool operator!= (const Accessors<T> & other) const
    {
        return this->data() != other.data();
    }

    __host__ __device__
    inline bool operator< (const T * other) const
    {
        return this->data() < other;
    }

    __host__ __device__
    inline bool operator< (const Accessors<T> & other) const
    {
        return this->data() < other.data();
    }

    __host__ __device__
    inline bool operator<= (const T * other) const
    {
        return this->data() <= other;
    }

    __host__ __device__
    inline bool operator<= (const Accessors<T> & other) const
    {
        return this->data() <= other.data();
    }

    __host__ __device__
    inline bool operator> (const T * other) const
    {
        return this->data() > other;
    }

    __host__ __device__
    inline bool operator> (const Accessors<T> & other) const
    {
        return this->data() > other.data();
    }

    __host__ __device__
    inline bool operator>= (const T * other) const
    {
        return this->data() >= other;
    }

    __host__ __device__
    inline bool operator>= (const Accessors<T> & other) const
    {
        return this->data() >= other.data();
    }

    //--- arithmetic ---------------------------------------------------

    __host__ __device__
    inline final_type & operator++()
    {
        this->data() += this->stride();
        return *this;
    }

    __host__ __device__
    inline final_type operator++(int)
    {
        final_type copy = *this;
        this->data() += this->stride();
        return copy;
    }

    __host__ __device__
    inline final_type & operator+=(offset_type offset)
    {
        this->data() += offset * this->stride();
        return *this;
    }

    __host__ __device__
    inline final_type & operator-=(offset_type offset)
    {
        this->data() -= offset * this->stride();
        return *this;
    }

    // --- external operators ---

    __host__ __device__
    friend inline final_type operator+(const final_type & ptr, offset_type offset)
    {
        auto copy = ptr;
        copy += offset;
        return copy;
    }

    __host__ __device__
    friend inline final_type operator-(const final_type & ptr, offset_type offset)
    {
        auto copy = ptr;
        copy -= offset;
        return copy;
    }
};

// =====================================================================
// 2. Implement iterators
// =====================================================================

// Main base class that deals with strides
template <typename T, long S = 1, typename D = void>
class Iterators: public Accessors<T, Iterators<T,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = Iterators<T, S, D>;
    using this_type_final = Iterators<T, S>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = Accessors<T, this_type>;
    using value_type = typename parent_type::value_type;
    using size_type = typename parent_type::size_type;
    using offset_type = typename parent_type::offset_type;
    using reference = typename parent_type::reference;
    using const_reference = typename parent_type::const_reference;
    using pointer = typename parent_type::pointer;
    using const_pointer = typename parent_type::const_pointer;

    // new
    static constexpr offset_type static_stride = S;
    using iterator = WeakRef<value_type, static_stride>;
    using const_iterator = WeakRef<const value_type, static_stride>;
    using reverse_iterator = WeakRef<value_type, -static_stride>;
    using const_reverse_iterator = WeakRef<const value_type, -static_stride>;


    //--- destructor ---------------------------------------------------

    __host__ __device__
    virtual ~Iterators() {}

    //--- implicit conversion ------------------------------------------

    __host__ __device__
    operator WeakRef<T, S> () const
    {
        return WeakRef<T, S>(this->data(), this->stride());
    }

//    operator WeakRef<const T, S> () const
//    {
//        return WeakRef<const T, S>(this->data(), this->stride());
//    }

    //--- concrete -----------------------------------------------------

    __host__ __device__
    inline offset_type stride() const
    {
        return S;
    }

    //--- iterators ----------------------------------------------------

    __host__ __device__
    inline iterator begin() const
    {
        return iterator(this->data(), this->stride());
    }

    __host__ __device__
    inline reverse_iterator rend() const
    {
        return reverse_iterator(
            this->data() - this->stride(), -this->stride());
    }

    __host__ __device__
    inline const_iterator cbegin() const
    {
        return this->begin();
    }

    __host__ __device__
    inline const_reverse_iterator crend() const
    {
        return this->rend();
    }
};

// =====================================================================
// 3. Static vs Dynamic stride
// =====================================================================

template <typename T, long S = 1, typename D = void>
class Impl: public Iterators<T, S, Impl<T,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = Impl<T,S,D>;
    using this_type_final = Impl<T,S>;
    using final_type = typename internal::guess_type<this_type_final,D>::value;

    // inherited
    using parent_type = Iterators<T, S, this_type>;
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
    __host__ __device__
    Impl(offset_type stride = S)
    {
        if (S != stride) throw std::runtime_error("stride not consistent");
    }

    //--- concrete -----------------------------------------------------

    __host__ __device__
    inline offset_type stride() const
    {
        return S;
    }
};

template <typename T, typename D>
class Impl<T, DynamicStride, D>: public Iterators<T, DynamicStride, Impl<T,DynamicStride,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = Impl<T,DynamicStride,D>;
    using this_type_final = Impl<T,DynamicStride>;
    using final_type = typename internal::guess_type<this_type_final,D>::value;

    // inherited
    using parent_type = Iterators<T, DynamicStride, this_type>;
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

    __host__ __device__
    Impl(offset_type stride): _stride(stride)
    {}

    //--- concrete -----------------------------------------------------

    __host__ __device__
    inline offset_type stride() const
    {
        return this->_stride;
    }

protected:
    offset_type _stride;
};


} // namespace abstractptr
} // namespace internal

// =====================================================================
// Final. Public class
// =====================================================================

template <typename T, long S, typename D>
class AbstractPointer:
public internal::abstractptr::Impl<T, S, AbstractPointer<T,S,D> >
{
public:

    //--- types --------------------------------------------------------

    using this_type = AbstractPointer<T,S,D>;
    using this_type_final = AbstractPointer<T,S>;
    using final_type = typename internal::guess_type<this_type_final,D>::value;

    // inherited
    using parent_type = internal::abstractptr::Impl<T, S, this_type>;
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
    virtual ~AbstractPointer() {}

    //--- constructor --------------------------------------------------

    using internal::abstractptr::Impl<T, S, AbstractPointer<T,S,D> >::Impl;
};

template <typename T, typename D=void>
using AbstractDynamicPointer = AbstractPointer<T, DynamicStride, D>;


} // namespace jf

#endif // JF_VECTOR_ABSTRACTPTR_H
