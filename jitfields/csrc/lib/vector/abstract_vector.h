#ifndef JF_VECTOR_ABTRACTVECTOR_H
#define JF_VECTOR_ABTRACTVECTOR_H
#include "forward.h"
#include "traits.h"
#include "weak_sized.h"
#include "../cuda_switch.h"
#include <tuple>

namespace jf {


namespace internal {
namespace abstractvec {

// =====================================================================
// 1. Implement accessors
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
    operator const_reference () const { return this->data(); }

    __host__ __device__
    operator reference () { return this->data(); }

    //--- virtual ------------------------------------------------------

    __host__ __device__
    virtual pointer data() = 0;

    __host__ __device__
    virtual const_pointer data() const = 0;

    __host__ __device__
    virtual offset_type stride() const = 0;

    __host__ __device__
    virtual size_type size() const = 0;

    //--- accessors ----------------------------------------------------

    __host__ __device__
    inline const T & x() const noexcept
    {
        return (*this)[0];
    }

    __host__ __device__
    inline const T & y() const noexcept
    {
        return (*this)[1];
    }

    __host__ __device__
    inline const T & z() const noexcept
    {
        return (*this)[2];
    }

    __host__ __device__
    inline T & x() noexcept
    {
        return (*this)[0];
    }

    __host__ __device__
    inline T & y() noexcept
    {
        return (*this)[1];
    }

    __host__ __device__
    inline T & z() noexcept
    {
        return (*this)[2];
    }

    __host__ __device__
    inline reference operator*() noexcept
    {
        return *(this->data());
    }

    __host__ __device__
    inline const_reference operator*() const noexcept
    {
        return *(this->data());
    }

    __host__ __device__
    inline reference operator[](offset_type i) noexcept
    {
        return this->data()[i*this->stride()];
    }

    __host__ __device__
    inline const_reference operator[](offset_type i) const  noexcept
    {
        return this->data()[i*this->stride()];
    }

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

    //--- setters ------------------------------------------------------

    __host__ __device__
    inline void fill(const T & value)
    {
        for (long d = 0; d < this->size(); ++d)
            (*this)[d] = value;
    }

    template <typename U>
    __host__ __device__
    inline void copy(const U * other) // assume stride 1 and length N
    {
        for (size_type d = 0; d < this->size(); ++d)
            (*this)[d] = static_cast<T>(this->fill[d]);
    }

    template <typename Iterator>
    __host__ __device__
    inline void copy(Iterator begin, const Iterator & end)
    {
        for (size_type d=0; begin != end; ++begin, ++d)
            (*this)[d] = static_cast<T>(*begin);
    }

    template <typename OT, unsigned long ON, long OS, typename OD>
    __host__ __device__
    inline void copy(const AbstractSizedPointer<OT,ON,OS,OD> & other)
    {
        size_type nb_elem = (this->size() >= other.size() ? this->size() : other.size());
        for (size_type d = 0; d < nb_elem; ++d)
            (*this)[d] = static_cast<T>(other[d]);
    }

    template <typename OT, unsigned long ON, long OS, typename OD>
    __host__ __device__
    inline void copy(const AbstractVector<OT,ON,OS,OD> & other)
    {
        size_type nb_elem = (this->size() >= other.size() ? this->size() : other.size());
        for (size_type d = 0; d < nb_elem; ++d)
            (*this)[d] = static_cast<T>(other[d]);
    }

    //--- comparisons --------------------------------------------------

    template <typename U>
    __host__ __device__
    inline bool operator == (const U & other)
    {
        for (size_type d=0; d<this->size(); ++d)
            if ((*this)[d] != other[d]) return false;
        return true;
    }

    template <typename U>
    __host__ __device__
    inline bool operator != (const U & other)
    {
        return !((*this) == other);
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

    //--- concrete -----------------------------------------------------

    __host__ __device__
    inline offset_type stride() const
    {
        return S;
    }

    //--- iterators ----------------------------------------------------

    __host__ __device__
    inline iterator begin()
    {
        return iterator(this->data(), this->stride());
    }

    __host__ __device__
    inline const_iterator begin() const
    {
        return iterator(this->data(), this->stride());
    }

    __host__ __device__
    inline reverse_iterator rend()
    {
        return reverse_iterator(
            this->data() - this->stride(), -this->stride());
    }

    __host__ __device__
    inline const_reverse_iterator rend() const
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

    __host__ __device__
    inline iterator end()
    {
        return iterator(
            this->data() + this->size() * this->stride(), this->stride());
    }

    __host__ __device__
    inline const_iterator end() const
    {
        return iterator(
            this->data() + this->size() * this->stride(), this->stride());
    }

    __host__ __device__
    inline reverse_iterator rbegin()
    {
        return reverse_iterator(
            this->data() + (this->size() - 1) * this->stride(),
            -this->stride());
    }

    __host__ __device__
    inline const_reverse_iterator rbegin() const
    {
        return reverse_iterator(
            this->data() + (this->size() - 1) * this->stride(),
            -this->stride());
    }

    __host__ __device__
    inline const_iterator cend() const
    {
        return this->end();
    }

    __host__ __device__
    inline const_reverse_iterator crbegin() const
    {
        return this->rbegin();
    }
};

// =====================================================================
// 3. Static vs Dynamic stride
// =====================================================================

template <typename T, long S = 1, typename D = void>
class SwitchStride: public Iterators<T, S, SwitchStride<T,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = SwitchStride<T,S,D>;
    using this_type_final = SwitchStride<T,S>;
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
    virtual ~SwitchStride() {}

    //--- constructor --------------------------------------------------

    __host__ __device__
    SwitchStride(offset_type stride = S)
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
class SwitchStride<T, DynamicStride, D>: public Iterators<T, DynamicStride, SwitchStride<T,DynamicStride,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = SwitchStride<T,DynamicStride,D>;
    using this_type_final = SwitchStride<T,DynamicStride>;
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
    virtual ~SwitchStride() {}

    //--- constructor --------------------------------------------------

    __host__ __device__
    SwitchStride(offset_type stride): _stride(stride)
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

// =====================================================================
// 4. Static vs Dynamic size
// =====================================================================

template <typename T, unsigned long N = DynamicSize, long S = 1, typename D = void>
class SwitchSize: public SwitchStride<T, S, SwitchSize<T,N,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = SwitchSize<T,N,S,D>;
    using this_type_final = SwitchSize<T,N,S>;
    using final_type = typename internal::guess_type<this_type_final,D>::value;

    // inherited
    using parent_type = SwitchStride<T, S, this_type>;
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
    virtual ~SwitchSize() {}

    //--- constructor --------------------------------------------------

    __host__ __device__
    SwitchSize(size_type size = N, offset_type stride = S):
        parent_type(stride)
    {
        if (N != size) throw std::runtime_error("size not consistent");
    }

    //--- concrete -----------------------------------------------------

    __host__ __device__
    inline size_type size() const
    {
        return N;
    }
};

template <typename T, long S, typename D>
class SwitchSize<T, DynamicSize, S, D>: public SwitchStride<T, S, SwitchSize<T,DynamicSize,S,D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = SwitchSize<T,DynamicSize,S,D>;
    using this_type_final = SwitchSize<T,DynamicSize,S>;
    using final_type = typename internal::guess_type<this_type_final,D>::value;

    // inherited
    using parent_type = SwitchStride<T, S, this_type>;
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
    virtual ~SwitchSize() {}

    //--- constructor --------------------------------------------------

    __host__ __device__
    SwitchSize(size_type size, offset_type stride=S):
        parent_type(stride), _size(size)
    {}

    //--- concrete -----------------------------------------------------

    __host__ __device__
    inline size_type size() const
    {
        return this->_size;
    }

protected:
    size_type _size;
};

} // namespace abstractvec
} // namespace internal

// =====================================================================
// 5. Public class
// =====================================================================

template <typename T, unsigned long N, long S, typename D>
class AbstractVector:
public internal::abstractvec::SwitchSize<T, N, S, AbstractVector<T,N,S,D> >
{
public:

    //--- types --------------------------------------------------------

    using this_type = AbstractVector<T, N, S, D>;
    using this_type_final = AbstractVector<T, N, S>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = internal::abstractvec::SwitchSize<T, N, S, this_type>;
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

    // new
    using weak_type = WeakVector<T, N, S>;
    using const_weak_type = WeakVector<const T, N, S>;
    using ref_type = WeakSizedRef<T, N, S>;
    using const_ref_type = WeakSizedRef<const T, N, S>;

    //--- destructor ---------------------------------------------------

    __host__ __device__
    virtual ~AbstractVector() {}

    //--- constructor ---------------------------------------------------

    using internal::abstractvec::SwitchSize<T, N, S, AbstractVector<T,N,S,D> >::SwitchSize;

    //--- conversions --------------------------------------------------

    __host__ __device__
    inline weak_type weak()
    {
        return weak_type(this->size(), this->data(), this->stride());
    }

    __host__ __device__
    inline const_weak_type weak() const
    {
        return const_weak_type(this->size(), this->data(), this->stride());
    }

    __host__ __device__
    inline const_weak_type cweak() const
    {
        return this->ref();
    }

    __host__ __device__
    inline ref_type ref()
    {
        return ref_type(this->size(), this->data(), this->stride());
    }

    __host__ __device__
    inline const_ref_type ref() const
    {
        return const_ref_type(this->size(), this->data(), this->stride());
    }

    __host__ __device__
    inline const_ref_type cref() const
    {
        return this->ref();
    }

    __host__ __device__
    inline operator weak_type ()
    {
        return this->weak();
    }

    __host__ __device__
    inline operator const_weak_type () const
    {
        return this->cweak();
    }

    __host__ __device__
    inline operator ref_type ()
    {
        return this->ref();
    }

    __host__ __device__
    inline operator const_ref_type () const
    {
        return this->cref();
    }

    //--- unbind -------------------------------------------------------

    template <typename... U>
    __host__ __device__
    void unbind(U&... x) const
    {
        return this->ref().unbind();
    }
};

// =====================================================================
//  Unpack API
// =====================================================================

template <unsigned long I, typename T, unsigned long N, long S, typename D>
__host__ __device__
inline T& get(AbstractVector<T, N, S, D> & v) noexcept
{
    return v[I];
};

template <unsigned long I, typename T, unsigned long N, long S, typename D>
__host__ __device__
inline T&& get(AbstractVector<T, N, S, D> && v) noexcept
{
    return static_cast<T&&>(v[I]);
};

template <unsigned long I, typename T, unsigned long N, long S, typename D>
__host__ __device__
inline const T& get(const AbstractVector<T, N, S, D> & v) noexcept
{
    return v[I];
};

template <unsigned long I, typename T, unsigned long N, long S, typename D>
__host__ __device__
inline const T&& get(const AbstractVector<T, N, S, D> && v) noexcept
{
    return static_cast<const T&&>(v[I]);
};

} // namespace jf

template <typename T, unsigned long N, long S, typename D>
struct std::tuple_size< jf::AbstractVector<T, N, S, D> >
{
    using value_type = std::size_t;
    static constexpr value_type value = static_cast<std::size_t>(N);
    using type = std::integral_constant<value_type, value>;
};

template <std::size_t I, typename T, unsigned long N, long S, typename D>
struct std::tuple_element< I, jf::AbstractVector<T, N, S, D> >
{
    using type = T;
};



#endif // JF_VECTOR_ABTRACTVECTOR_H
