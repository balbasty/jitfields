#ifndef JF_DISTANCE_MESH_UTILS_H
#define JF_DISTANCE_MESH_UTILS_H
#include "../utils.h"

// =============================================================================
//
//                          VECTOR MATH HELPERS
//
// =============================================================================

namespace jf {
namespace distance_mesh {

// =============================================================================
//     POINTERS
// =============================================================================

template <typename value_t, typename offset_t = long>
struct StridedPointer {
    using value_type  = value_t;
    using offset_type = offset_t;

    __host__ __device__ StridedPointer(value_t * ptr, offset_t stride):
        ptr(ptr), stride(stride) {}

    __host__ __device__ inline value_t & operator[] (offset_t n) { return ptr[n*stride]; }
    __host__ __device__ inline const value_t & operator[] (offset_t n) const { return ptr[n*stride]; }

    value_t * ptr;
    offset_t stride;
};

template <typename value_t, typename offset_t = long>
struct SizedStridedPointer {
    using value_type  = value_t;
    using offset_type = offset_t;

    __host__ __device__ SizedStridedPointer(value_t * ptr, offset_t stride, offset_t size):
        ptr(ptr), stride(stride), size(size) {}

    __host__ __device__ inline value_t & operator[] (offset_t n) { return ptr[n*stride]; }
    __host__ __device__ inline const value_t & operator[] (offset_t n) const { return ptr[n*stride]; }

    value_t * ptr;
    offset_t stride;
    offset_t size;
};

// =============================================================================
//     1D VECTORS
// =============================================================================

template <typename offset_t = long>
struct Sized {
    using offset_type = offset_t;

    virtual ~Sized() {}

    __host__ __device__ Sized(offset_t length): length(length) {}

    __host__ __device__ inline offset_t size() const { return length; }

    offset_t length;
};

template <long N, typename offset_t = long>
struct StaticSized {
    using offset_type = offset_t;

    static constexpr offset_t length = N;

    virtual ~StaticSized() {}

    __host__ __device__ inline offset_t size() const { return length; }
};


template <long D, typename scalar_t>
struct AnyConstPoint {
    using value_type = scalar_t;

    virtual ~AnyConstPoint() {}

    __host__ __device__ virtual const scalar_t& operator[] (long d) const = 0;
};

template <long D, typename scalar_t>
struct AnyPoint {
    using value_type = scalar_t;

    virtual ~AnyPoint() {}

    __host__ __device__ virtual scalar_t& operator[] (long d) = 0;
};

template <long D, typename scalar_t>
struct StaticPoint;
template <long D, typename scalar_t>
struct RefPoint;
template <long D, typename scalar_t>
struct ConstRefPoint;
template <long D, typename scalar_t, typename offset_t = long>
struct StridedPoint;
template <long D, typename scalar_t, typename offset_t = long>
struct ConstStridedPoint;


template <long D, typename scalar_t, typename FinalType = void>
struct PointMixin: public AnyPoint<D, scalar_t> {

    using this_type         = PointMixin<D, scalar_t, FinalType>;
    using final_type        = FinalType;
    using static_type       = StaticPoint<D, scalar_t>;
    using point_type        = AnyPoint<D, scalar_t>;
    using const_point_type  = AnyConstPoint<D, scalar_t>;
    using value_type        = scalar_t;

    virtual ~PointMixin() {}

    // reference to final type

    __host__ __device__ inline 
    final_type * thisptr() { return reinterpret_cast<final_type*>(this); }
    __host__ __device__ inline 
    const final_type * thisptr() const { return reinterpret_cast<const final_type*>(this); }
    __host__ __device__ inline 
    final_type & thisref() { return reinterpret_cast<final_type&>(*this); }
    __host__ __device__ inline 
    const final_type & thisref() const { return reinterpret_cast<const final_type&>(*this); }

    // in-place operations

    __host__ __device__ inline
    final_type& copy_ (const const_point_type & other)
    { for (long d=0; d < D; ++d) (*this)[d] = other[d]; return thisref(); }

    __host__ __device__ inline
    final_type& copy_ (const const_point_type & other, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] = other[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& copy_ (scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] = alpha; return thisref(); }

    __host__ __device__ inline
    final_type& operator = (const const_point_type & other)
    { return this->copy_(other); }

    __host__ __device__ inline
    final_type& operator = (scalar_t alpha)
    { return this->copy_(alpha); }

    __host__ __device__ inline
    final_type& add_ (const const_point_type & other)
    { for (long d=0; d < D; ++d) (*this)[d] += other[d]; return thisref(); }

    __host__ __device__ inline
    final_type& add_ (const const_point_type & other, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] += other[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& add_ (scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] += alpha; return thisref(); }

    __host__ __device__ inline
    final_type& operator += (const const_point_type & other)
    { return this->add_(other); }

    __host__ __device__ inline
    final_type& operator += (scalar_t alpha)
    { return this->add_(alpha); }

    __host__ __device__ inline
    final_type& sub_ (const const_point_type & other)
    { for (long d=0; d < D; ++d) (*this)[d] -= other[d]; return thisref(); }

    __host__ __device__ inline
    final_type& sub_ (const const_point_type & other, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] -= other[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& sub_ (scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] -= alpha; return thisref(); }

    __host__ __device__ inline
    final_type& operator -= (const const_point_type & other)
    { return this->sub_(other); }

    __host__ __device__ inline
    final_type& operator -= (scalar_t alpha)
    { return this->sub_(alpha); }

    __host__ __device__ inline
    final_type& mul_ (const const_point_type & other)
    { for (long d=0; d < D; ++d) (*this)[d] *= other[d]; return thisref(); }

    __host__ __device__ inline
    final_type& mul_ (const const_point_type & other, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] *= other[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& mul_ (scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] *= alpha; return thisref(); }

    __host__ __device__ inline
    final_type& operator *= (const const_point_type & other)
    { return this->mul_(other); }

    __host__ __device__ inline
    final_type& operator *= (scalar_t alpha)
    { return this->mul_(alpha); }

    __host__ __device__ inline
    final_type& div_ (const const_point_type & other)
    { for (long d=0; d < D; ++d) (*this)[d] /= other[d]; return thisref(); }

    __host__ __device__ inline
    final_type& div_ (const const_point_type & other, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] /= other[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& div_ (scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] /= alpha; return thisref(); }

    __host__ __device__ inline
    final_type& operator /= (const const_point_type & other)
    { return this->div_(other); }

    __host__ __device__ inline
    final_type& operator /= (scalar_t alpha)
    { return this->div_(alpha); }

    __host__ __device__ inline
    final_type& max_(const const_point_type & other)
    { for (long d=0; d < D; ++d) (*this)[d] = jf::max((*this)[d], other[d]); return thisref(); }

    __host__ __device__ inline
    final_type& min_(const const_point_type & other)
    { for (long d=0; d < D; ++d) (*this)[d] = jf::min((*this)[d], other[d]); return thisref(); }

    __host__ __device__ inline
    final_type& normalize_()
    { 
        scalar_t nrm = static_cast<scalar_t>(0); 
        for (long d=0; d < D; ++d) nrm += (*this)[d] * (*this)[d]; 
        nrm = sqrt(nrm);
        for (long d=0; d < D; ++d) (*this)[d] /= nrm; 
        return thisref(); 
      }


    // out-of-place operations (fill self)

    __host__ __device__ inline
    final_type& addto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (long d=0; d < D; ++d) (*this)[d] = lhs[d] + rhs[d]; return thisref(); }

    __host__ __device__ inline
    final_type& addto_(const const_point_type & lhs, const const_point_type & rhs, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] = lhs[d] + rhs[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& subto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (long d=0; d < D; ++d) (*this)[d] = lhs[d] - rhs[d]; return thisref(); }

    __host__ __device__ inline
    final_type& subto_(const const_point_type & lhs, const const_point_type & rhs, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] = lhs[d] - rhs[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& multo_(const const_point_type & lhs, const const_point_type & rhs)
    { for (long d=0; d < D; ++d) (*this)[d] = lhs[d] * rhs[d]; return thisref(); }

    __host__ __device__ inline
    final_type& multo_(const const_point_type & lhs, const const_point_type & rhs, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] = lhs[d] * rhs[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& divto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (long d=0; d < D; ++d) (*this)[d] = lhs[d] / rhs[d]; return thisref(); }

    __host__ __device__ inline
    final_type& divto_(const const_point_type & lhs, const const_point_type & rhs, scalar_t alpha)
    { for (long d=0; d < D; ++d) (*this)[d] = lhs[d] / rhs[d] * alpha; return thisref(); }

    __host__ __device__ inline
    final_type& maxto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (long d=0; d < D; ++d) (*this)[d] = jf::max(lhs[d], rhs[d]); return thisref(); }

    __host__ __device__ inline
    final_type& minto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (long d=0; d < D; ++d) (*this)[d] = jf::min(lhs[d], rhs[d]); return thisref(); }

    __host__ __device__ inline
    final_type& crossto_(const const_point_type & lhs, const const_point_type & rhs)
    {
        // !! only works in 3D
        (*this)[0] =  lhs[1]*rhs[2] - lhs[2]*rhs[1];
        (*this)[1] = -lhs[0]*rhs[2] + lhs[2]*rhs[0];
        (*this)[2] =  lhs[0]*rhs[1] - lhs[1]*rhs[0];
        return thisref();
    }

};


template <long D, typename scalar_t, typename FinalType = void>
struct ConstPointMixin: public AnyConstPoint<D, scalar_t> {

    using this_type         = PointMixin<D, scalar_t, FinalType>;
    using final_type        = FinalType;
    using static_type       = StaticPoint<D, scalar_t>;
    using point_type        = AnyPoint<D, scalar_t>;
    using const_point_type  = AnyConstPoint<D, scalar_t>;
    using value_type        = scalar_t;

    virtual ~ConstPointMixin() {}

    // out-of-place operations (return static point)

    __host__ __device__ inline
    static_type copy () const
    { return static_type(*this); }

    __host__ __device__ inline
    static_type add(const const_point_type & other) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] + other[d]; return out; }

    __host__ __device__ inline
    static_type add(const const_point_type & other, scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] + other[d] * alpha; return out; }

    __host__ __device__ inline
    static_type add(scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] + alpha; return out; }

    __host__ __device__ inline
    static_type operator+(const const_point_type & rhs) const
    { return this->add(rhs); }

    __host__ __device__ inline
    static_type operator+(scalar_t alpha) const
    { return this->add(alpha); }

    __host__ __device__ inline
    static_type sub(const const_point_type & other) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] - other[d]; return out; }

    __host__ __device__ inline
    static_type sub(const const_point_type & other, scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] - other[d] * alpha; return out; }

    __host__ __device__ inline
    static_type sub(scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] - alpha; return out; }

    __host__ __device__ inline
    static_type operator-(const const_point_type & rhs) const
    { return this->sub(rhs); }

    __host__ __device__ inline
    static_type operator-(scalar_t alpha) const
    { return this->sub(alpha); }

    __host__ __device__ inline
    static_type mul(const const_point_type & other) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] * other[d]; return out; }

    __host__ __device__ inline
    static_type mul(const const_point_type & other, scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] * other[d] * alpha; return out; }

    __host__ __device__ inline
    static_type mul(scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] * alpha; return out; }

    __host__ __device__ inline
    static_type operator*(const const_point_type & rhs) const
    { return this->mul(rhs); }

    __host__ __device__ inline
    static_type operator*(scalar_t alpha) const
    { return this->mul(alpha); }

    __host__ __device__ inline
    static_type div(const const_point_type & other) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] / other[d]; return out; }

    __host__ __device__ inline
    static_type div(const const_point_type & other, scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] / (other[d] * alpha); return out; }

    __host__ __device__ inline
    static_type div(scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = (*this)[d] / alpha; return out; }

    __host__ __device__ inline
    static_type operator/(const const_point_type & rhs) const
    { return this->div(rhs); }

    __host__ __device__ inline
    static_type operator/(scalar_t alpha) const
    { return this->div(alpha); }

    __host__ __device__ inline
    static_type max(const const_point_type & other) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = jf::max((*this)[d], other[d]); return out; }

    __host__ __device__ inline
    static_type max(scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = jf::max((*this)[d], alpha); return out; }

    __host__ __device__ inline
    static_type min(const const_point_type & other) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = jf::min((*this)[d], other[d]); return out; }

    __host__ __device__ inline
    static_type min(scalar_t alpha) const
    { static_type out; for (long d=0; d < D; ++d) out[d] = jf::min((*this)[d], alpha); return out; }

    __host__ __device__ inline
    void cross(const const_point_type & other) const
    {
        // !! only works in 3D
        static_type out;
        out[0] =  (*this)[1]*other[2] - (*this)[2]*other[1];
        out[1] = -(*this)[0]*other[2] + (*this)[2]*other[0];
        out[2] =  (*this)[0]*other[1] - (*this)[1]*other[0];
        return out;
    }

    // operations that return a scalar

    __host__ __device__ inline
    scalar_t dot(const const_point_type & other) const
    { scalar_t out = static_cast<scalar_t>(0); for (long d=0; d < D; ++d) out += (*this)[d] * other[d]; return out; }

    __host__ __device__ inline
    scalar_t sum() const
    { scalar_t out = static_cast<scalar_t>(0); for (long d=0; d < D; ++d) out += (*this)[d]; return out; }

    __host__ __device__ inline
    scalar_t prod() const
    { scalar_t out = static_cast<scalar_t>(1); for (long d=0; d < D; ++d) out *= (*this)[d]; return out; }

    __host__ __device__ inline
    scalar_t sqnorm() const
    { return this->dot(*this); }

    __host__ __device__ inline
    scalar_t norm() const
    { return sqrt(this->sqnorm()); }

    // view

    template <long begin, long end>
    __host__ __device__ inline
    StaticPoint<end-begin, scalar_t> copyview()
    {
        StaticPoint<end-begin, scalar_t> out;
        for (long d=0; d<(end-begin); ++d) 
            out[d] = (*this)[begin+d];
        return out;
    }
};

template <long D, typename scalar_t>
struct StaticPoint: 
    public PointMixin <D, scalar_t, StaticPoint <D, scalar_t> >, 
    public ConstPointMixin <D, scalar_t, StaticPoint <D, scalar_t> >
{
    using this_type         = StaticPoint<D, scalar_t>;
    using const_point_type  = AnyConstPoint<D, scalar_t>;
    using value_type        = scalar_t;

    virtual ~StaticPoint() {}
    __host__ __device__ StaticPoint() = default;
    __host__ __device__ StaticPoint(const const_point_type & other)
    { this->copy_(other); }

    __host__ __device__ inline scalar_t& operator[] (long d) { return data[d]; };
    __host__ __device__ inline const scalar_t& operator[] (long d) const { return data[d]; };

    scalar_t data[D];

    // reference view

    template <long begin, long end>
    __host__ __device__ inline
    RefPoint<end-begin, scalar_t> view()
    {
        return RefPoint<end-begin, scalar_t>(data + begin);
    }

    template <long begin, long end>
    __host__ __device__ inline
    ConstRefPoint<end-begin, scalar_t> view() const
    {
        return ConstRefPoint<end-begin, scalar_t>(data + begin);
    }
    
};

template <long D, typename scalar_t>
struct RefPoint: 
    public PointMixin <D, scalar_t, RefPoint <D, scalar_t> >,
    public ConstPointMixin <D, scalar_t, RefPoint <D, scalar_t> >
{
    using this_type         = RefPoint<D, scalar_t>;
    using value_type        = scalar_t;

    virtual ~RefPoint() {}
    __host__ __device__ RefPoint(scalar_t * data): data(data) {}

    __host__ __device__ inline scalar_t& operator[] (long d) { return data[d]; };
    __host__ __device__ inline const scalar_t& operator[] (long d) const { return data[d]; };

    scalar_t * data;


    // reference view

    template <long begin, long end>
    __host__ __device__ inline
    RefPoint<end-begin, scalar_t> view()
    {
        return RefPoint<end-begin, scalar_t>(data + begin);
    }

    template <long begin, long end>
    __host__ __device__ inline
    ConstRefPoint<end-begin, scalar_t> view() const
    {
        return ConstRefPoint<end-begin, scalar_t>(data + begin);
    }
};

template <long D, typename scalar_t>
struct ConstRefPoint: 
    public ConstPointMixin <D, scalar_t, ConstRefPoint<D, scalar_t> >
{
    using this_type         = ConstRefPoint<D, scalar_t>;
    using value_type        = scalar_t;

    virtual ~ConstRefPoint() {}
    __host__ __device__ ConstRefPoint(const scalar_t * data): data(data) {}

    __host__ __device__ inline const scalar_t& operator[] (long d) const { return data[d]; };

    const scalar_t * data;

    // reference view

    template <long begin, long end>
    __host__ __device__ inline
    ConstRefPoint<end-begin, scalar_t> view() const
    {
        return ConstRefPoint<end-begin, scalar_t>(data + begin);
    }
};

template <long D, typename scalar_t, typename offset_t>
struct StridedPoint: 
    public PointMixin <D, scalar_t, StridedPoint<D, scalar_t, offset_t> >,
    public ConstPointMixin <D, scalar_t, StridedPoint<D, scalar_t, offset_t> >
{
    using this_type         = StridedPoint<D, scalar_t, offset_t>;
    using const_point_type  = AnyConstPoint<D, scalar_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;

    virtual ~StridedPoint() {}
    __host__ __device__ StridedPoint(scalar_t * data, offset_t stride): data(data), stride(stride) {}

    __host__ __device__ inline scalar_t& operator[] (long d) { return data[d*stride]; };
    __host__ __device__ inline const scalar_t& operator[] (long d) const { return data[d*stride]; };

    __host__ __device__ inline this_type & operator= (const const_point_type & other)
    {
        for (offset_t d=0; d<D; ++d) (*this)[d] = other[d];
        return *this;
    }

    scalar_t * data;
    offset_t stride;

    // reference view

    template <long begin, long end>
    __host__ __device__ inline
    StridedPoint<end-begin, scalar_t, offset_t> view()
    {
        return StridedPoint<end-begin, scalar_t, offset_t>(data + begin, stride);
    }

    template <long begin, long end>
    __host__ __device__ inline
    ConstStridedPoint<end-begin, scalar_t, offset_t> view() const
    {
        return ConstStridedPoint<end-begin, scalar_t, offset_t>(data + begin, stride);
    }
};

template <long D, typename scalar_t, typename offset_t>
struct ConstStridedPoint: 
    public ConstPointMixin <D, scalar_t, ConstStridedPoint<D, scalar_t, offset_t> >
{
    using this_type         = ConstStridedPoint<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;

    virtual ~ConstStridedPoint() {}
    __host__ __device__ ConstStridedPoint(const scalar_t * data, offset_t stride): 
        data(data), stride(stride) {}

    __host__ __device__ inline const scalar_t& operator[] (long d) const 
    { return data[d*stride]; };

    const scalar_t * data;
    offset_t stride;

    // reference view

    template <long begin, long end>
    __host__ __device__ inline
    const ConstStridedPoint<end-begin, scalar_t, offset_t> view() const
    {
        return ConstStridedPoint<end-begin, scalar_t, offset_t>(data + begin, stride);
    }
};

// =============================================================================
//     VECTORS OF VECTORS
// =============================================================================

template <long N, long D, typename scalar_t>
struct StaticPointList: public StaticSized<N> {

    using this_type         = StaticPointList<N, D, scalar_t>;
    using value_type        = scalar_t;
    using PointType         = RefPoint<D, scalar_t>;
    using ConstPointType    = ConstRefPoint<D, scalar_t>;

    virtual ~StaticPointList() {}

    __host__ __device__ inline long size() const { return N; }

    __host__ __device__ inline PointType operator[] (long n) 
    { return PointType(data + n*D); };
    __host__ __device__ inline ConstPointType operator[] (long n)  const
    { return ConstPointType(data + n*D); };

    scalar_t data[N*D];
};

template <long D, typename scalar_t>
struct RefPointList {

    using this_type         = RefPointList<D, scalar_t>;
    using value_type        = scalar_t;
    using PointType         = RefPoint<D, scalar_t>;
    using ConstPointType    = ConstRefPoint<D, scalar_t>;

    virtual ~RefPointList() {}
    __host__ __device__ RefPointList(scalar_t * data): data(data) {}

    __host__ __device__ inline PointType operator[] (long n) 
    { return PointType(data + n*D); };
    __host__ __device__ inline ConstPointType operator[] (long n)  const
    { return ConstPointType(data + n*D); };

    scalar_t * data = nullptr;
};

template <long D, typename scalar_t, typename offset_t = long>
struct RefPointListSized: 
    public RefPointList<D, scalar_t>,
    public Sized<offset_t>
{
    using this_type         = RefPointListSized<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;
    using BaseList          = RefPointList<D, scalar_t>;
    using BaseSized         = Sized<offset_t>;

    virtual ~RefPointListSized() {}
    __host__ __device__ RefPointListSized(scalar_t * data, offset_t length): 
        BaseList(data), BaseSized(length) {}
};

template <long D, typename scalar_t>
struct ConstRefPointList {

    using this_type         = ConstRefPointList<D, scalar_t>;
    using value_type        = scalar_t;
    using ConstPointType    = ConstRefPoint<D, scalar_t>;

    virtual ~ConstRefPointList() {}
    __host__ __device__ ConstRefPointList(const scalar_t * data): data(data) {}

    __host__ __device__ inline ConstPointType operator[] (long n)  const
    { return ConstPointType(data + n*D); };

    const scalar_t * data = nullptr;
};

template <long D, typename scalar_t, typename offset_t = long>
struct ConstRefPointListSized: 
    public ConstRefPointList<D, scalar_t>,
    public Sized<offset_t>
{
    using this_type         = ConstRefPointListSized<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;
    using BaseList          = ConstRefPointList<D, scalar_t>;
    using BaseSized         = Sized<offset_t>;

    virtual ~ConstRefPointListSized() {}
    __host__ __device__ ConstRefPointListSized(const scalar_t * data, offset_t length): 
        BaseList(data), BaseSized(length) {}

};

template <long D, typename scalar_t, typename offset_t = long>
struct StridedPointList {

    using this_type         = StridedPointList<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;
    using PointType         = StridedPoint<D, scalar_t, offset_t>;
    using ConstPointType    = ConstStridedPoint<D, scalar_t, offset_t>;

    virtual ~StridedPointList() {}
    __host__ __device__
    StridedPointList(scalar_t * data, 
                     offset_t stride_elem, 
                     offset_t stride_channel): 
        data(data), stride_elem(stride_elem), stride_channel(stride_channel) {}

    __host__ __device__ inline PointType operator[] (long n) 
    { return PointType(data + n*stride_elem, stride_channel); };

    __host__ __device__ inline ConstPointType operator[] (long n)  const
    { return ConstPointType(data + n*stride_elem, stride_channel); };

    scalar_t * data         = nullptr;
    offset_t stride_elem    = static_cast<offset_t>(1);
    offset_t stride_channel = static_cast<offset_t>(1);
};

template <long D, typename scalar_t, typename offset_t = long>
struct StridedPointListSized: 
    public StridedPointList<D, scalar_t, offset_t>,
    public Sized<offset_t>
{
    using this_type         = StridedPointListSized<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;
    using BaseList          = StridedPointList<D, scalar_t, offset_t>;
    using BaseSized         = Sized<offset_t>;

    virtual ~StridedPointListSized() {}
    __host__ __device__ StridedPointListSized(scalar_t * data, 
                     offset_t stride_elem, 
                     offset_t stride_channel,
                     offset_t length): 
        BaseList(data, stride_elem, stride_channel), BaseSized(length) {}
};

template <long D, typename scalar_t, typename offset_t = long>
struct ConstStridedPointList {

    using this_type         = ConstStridedPointList<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;
    using ConstPointType    = ConstStridedPoint<D, scalar_t, offset_t>;

    virtual ~ConstStridedPointList() {}
    __host__ __device__
    ConstStridedPointList(const scalar_t * data, 
                          offset_t stride_elem, 
                          offset_t stride_channel): 
        data(data), stride_elem(stride_elem), stride_channel(stride_channel) {}

    __host__ __device__ inline ConstPointType operator[] (long n)  const
    { return ConstPointType(data + n*stride_elem, stride_channel); };

    const scalar_t * data   = nullptr;
    offset_t stride_elem    = static_cast<offset_t>(1);
    offset_t stride_channel = static_cast<offset_t>(1);
};

template <long D, typename scalar_t, typename offset_t = long>
struct ConstStridedPointListSized: 
    public ConstStridedPointList<D, scalar_t, offset_t>,
    public Sized<offset_t>
{
    using this_type         = ConstStridedPointListSized<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;
    using BaseList          = ConstStridedPointList<D, scalar_t, offset_t>;
    using BaseSized         = Sized<offset_t>;

    virtual ~ConstStridedPointListSized() {}
    __host__ __device__ 
    ConstStridedPointListSized(const scalar_t * data, 
                               offset_t stride_elem, 
                               offset_t stride_channel,
                               offset_t length): 
        BaseList(data, stride_elem, stride_channel), BaseSized(length) {}
};


// =============================================================================
//     ARRAYS OF VECTORS
// =============================================================================

template <long... N>
struct _Prod {};
template <long N0, long... N>
struct _Prod<N0, N...> { static constexpr long value = N0 * _Prod<N...>::value; };
template <long N0>
struct _Prod<N0> { static constexpr long value = N0; };
template <>
struct _Prod<> { static constexpr long value = 1; };

template <typename... N>
struct _Count {};
template <typename N0, typename... N>
struct _Count<N0, N...> { static constexpr long value = 1 + _Count<N...>::value; };
template <typename N0>
struct _Count<N0> { static constexpr long value = 1; };
template <>
struct _Count<> { static constexpr long value = 0; };

template <long... N>
struct _CountInt {};
template <long N0, long... N>
struct _CountInt<N0, N...> { static constexpr long value = 1 + _CountInt<N...>::value; };
template <long N0>
struct _CountInt<N0> { static constexpr long value = 1; };
template <>
struct _CountInt<> { static constexpr long value = 0; };

template <long D, typename scalar_t, long N0, long... N>
struct StaticPointArray {
    using this_type         = StaticPointArray<D, scalar_t, N0, N...>;
    using value_type        = scalar_t;
    using offset_type       = long;

    using PointType    = StaticPoint<D, scalar_t>;
    using SubArrayType = StaticPointArray<D, scalar_t, N...>;
    static constexpr long stride0 = _Prod<N...>::value * D;
    static constexpr long nbatch  = _CountInt<N...>::value + 1;

    template <long COUNT, bool dummy=true> // need dummy parameter to avoid explicit specialization
    struct returned { using type = typename SubArrayType::template returned<COUNT-1>::type; };
    template <bool dummy>
    struct returned<nbatch-1, dummy> { using type = PointType; };
    template <bool dummy>
    struct returned<0, dummy> { using type = SubArrayType; };

    virtual ~StaticPointArray() {}

    template <typename... T>
    __host__ __device__  inline 
    typename returned<_Count<T...>::value>::type & at(offset_type n0, T... n) 
    { 
        return (*this)[n0].at(n...);
    };
    __host__ __device__ inline 
    SubArrayType & at (offset_type n0) 
    { 
        return reinterpret_cast<SubArrayType&>(data + n0 * stride0);
    };
    __host__ __device__ inline 
    SubArrayType & operator[] (offset_type n0) 
    { 
        return this->at(n0);
    };

    template <typename... T>
    __host__ __device__  inline 
    const typename returned<_Count<T...>::value>::type & at(offset_type n0, T... n) const
    { 
        return (*this)[n0].at(n...);
    };
    __host__ __device__ inline 
    const SubArrayType& at(offset_type n0) const
    { 
        return reinterpret_cast<const SubArrayType&>(data + n0 * stride0);
    };
    __host__ __device__ inline 
    const SubArrayType& operator[] (offset_type n0) const
    { 
        return this->at(n0);
    };

    scalar_t data[N0*stride0];
};

template <long D, typename scalar_t, long N0>
struct StaticPointArray<D, scalar_t, N0> {

    using this_type         = StaticPointArray<D, scalar_t, N0>;
    using value_type        = scalar_t;
    using offset_type       = long;

    using PointType = StaticPoint<D, scalar_t>;

    template <long COUNT>
    struct returned { using type = PointType; };

    virtual ~StaticPointArray() {}

    __host__ __device__ inline 
    PointType& at (offset_type n0) 
    { 
        return reinterpret_cast<PointType&>(data + n0 * D);
    };
    __host__ __device__ inline 
    PointType& operator[] (offset_type n0) 
    { 
        return this->at(n0);
    };

    __host__ __device__ inline 
    const PointType& at (offset_type n0) const
    { 
        return reinterpret_cast<const PointType&>(data + n0 * D);
    };
    __host__ __device__ inline 
    const PointType& operator[] (offset_type n0) const
    { 
        return this->at(n0);
    };

    scalar_t data[N0*D];
};

template <long D, typename scalar_t, long... N>
struct RefPointArray {};

template <long D, typename scalar_t, long N1, long... N>
struct RefPointArray<D, scalar_t, N1, N...> {

    using this_type         = RefPointArray<D, scalar_t, N1, N...>;
    using value_type        = scalar_t;
    using offset_type       = long;

    using PointType    = StaticPoint<D, scalar_t>;
    using SubArrayType = RefPointArray<D, scalar_t, N...>;
    static constexpr long stride0 = _Prod<N...>::value * D * N1;
    static constexpr long nbatch  = _CountInt<N...>::value + 2;

    template <long COUNT, bool dummy = true>
    struct returned { using type = typename SubArrayType::template returned<COUNT-1>::type; };
    template <bool dummy>
    struct returned<nbatch-1, dummy> { using type = PointType; };
    template <bool dummy>
    struct returned<0, dummy> { using type = SubArrayType; };

    virtual ~RefPointArray() {}

    template <typename... T>
    __host__ __device__  inline 
    typename returned<_Count<T...>::value>::type & at(offset_type n0, T... n) 
    { 
        return (*this)[n0].at(n...);
    };
    __host__ __device__ inline 
    SubArrayType & at(offset_type n0) 
    { 
        return reinterpret_cast<SubArrayType&>(data + n0 * stride0);
    };
    __host__ __device__ inline 
    SubArrayType & operator[](offset_type n0) 
    { 
        return this->at(n0);
    };


    template <typename... T>
    __host__ __device__  inline 
    const typename returned<_Count<T...>::value>::type & at(offset_type n0, T... n) const
    { 
        return (*this)[n0].at(n...);
    };
    __host__ __device__ inline 
    const SubArrayType& at(offset_type n0) const
    { 
        return reinterpret_cast<const SubArrayType&>(data + n0 * stride0);
    };
    __host__ __device__ inline 
    const SubArrayType& operator[] (offset_type n0) const
    { 
        return this->at(n0);
    };

    scalar_t * data;
};

template <long D, typename scalar_t>
struct RefPointArray<D, scalar_t> {

    using this_type         = RefPointArray<D, scalar_t>;
    using value_type        = scalar_t;
    using offset_type       = long;

    using PointType = StaticPoint<D, scalar_t>;

    template <long COUNT>
    struct returned { using type = PointType; };

    virtual ~RefPointArray() {}

    __host__ __device__ inline 
    PointType& at(offset_type n0) 
    { 
        return reinterpret_cast<PointType&>(data + n0 * D);
    };
    __host__ __device__ inline 
    PointType& operator[] (offset_type n0) 
    { 
        return this->at(n0);
    };

    __host__ __device__ inline 
    const PointType& at(offset_type n0) const
    { 
        return reinterpret_cast<const PointType&>(data + n0 * D);
    };
    __host__ __device__ inline 
    const PointType& operator[] (offset_type n0) const
    { 
        return this->at(n0);
    };

    scalar_t * data;
};

template <long D, typename scalar_t, typename offset_t, long... N>
struct StridedPointArray {};

template <long D, typename scalar_t, typename offset_t, long N1, long... N>
struct StridedPointArray<D, scalar_t, offset_t, N1, N...> {

    using this_type         = StridedPointArray<D, scalar_t, offset_t, N1, N...>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;

    using PointType    = StaticPoint<D, scalar_t>;
    using SubArrayType = StridedPointArray<D, scalar_t, offset_t, N...>;
    static constexpr long stride0 = _Prod<N...>::value * D * N1;
    static constexpr long nbatch  = _CountInt<N...>::value + 2;

    template <long COUNT, bool dummy = true>
    struct returned { using type = typename SubArrayType::template returned<COUNT-1>::type; };
    template <bool dummy>
    struct returned<nbatch-1, dummy> { using type = PointType; };
    template <bool dummy>
    struct returned<0, dummy> { using type = SubArrayType; };

    virtual ~StridedPointArray() {}

    template <typename Stride>
    __host__ __device__ 
    StridedPointArray(scalar_t * data, const Stride & stride): 
        data(data), stride(stride) {}

    __host__ __device__ 
    StridedPointArray(scalar_t * data = nullptr): 
        data(data), stride() { stride.copy_(1); }

    template <typename... T>
    __host__ __device__  inline 
    typename returned<_Count<T...>::value>::type at(offset_type n0, T... n) 
    { 
        return (*this)[n0].at(n...);
    };
    __host__ __device__ inline 
    SubArrayType at(offset_type n0) 
    { 
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    __host__ __device__ inline 
    SubArrayType operator[] (offset_type n0) 
    { 
        return this->at(n0);
    };

    template <typename... T>
    __host__ __device__  inline 
    const typename returned<_Count<T...>::value>::type at(offset_type n0, T... n) const
    { 
        return (*this)[n0].at(n...);
    };
    __host__ __device__ inline 
    const SubArrayType at(offset_type n0) const
    { 
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    __host__ __device__ inline 
    const SubArrayType operator[] (offset_type n0) const
    { 
        return this->at(n0);
    };

    scalar_t * data;
    StaticPoint<nbatch+1, offset_t> stride;
};

template <long D, typename scalar_t, typename offset_t>
struct StridedPointArray<D, scalar_t, offset_t> {

    using this_type         = StridedPointArray<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;

    using PointType = StridedPoint<D, scalar_t, offset_t>;
    using ConstPointType = StridedPoint<D, const scalar_t, offset_t>;

    virtual ~StridedPointArray() {}
    template <typename Stride>
    __host__ __device__ 
    StridedPointArray(scalar_t * data, const Stride & stride): 
        data(data), stride(stride) {}

    __host__ __device__  inline PointType at(offset_type n) 
    { 
        return PointType(data + n*stride[0], stride[1]); 
    };
    __host__ __device__  inline PointType operator[] (offset_type n) 
    { 
        return this->at(n);
    };

    __host__ __device__  inline ConstPointType at(offset_type n) const
    { 
        return ConstPointType(data + n*stride[0], stride[1]); 
    };
    __host__ __device__  inline ConstPointType operator[] (offset_type n) const
    { 
        return this->at(n);
    };

    scalar_t * data;
    StaticPoint<2, offset_t> stride;
};

template <long D, typename scalar_t, typename offset_t, long... N>
struct ConstStridedPointArray {};

template <long D, typename scalar_t, typename offset_t, long N1, long... N>
struct ConstStridedPointArray<D, scalar_t, offset_t, N1, N...> {

    using this_type         = ConstStridedPointArray<D, scalar_t, offset_t, N1, N...>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;

    using PointType    = StaticPoint<D, scalar_t>;
    using SubArrayType = ConstStridedPointArray<D, scalar_t, offset_t, N...>;
    static constexpr long stride0 = _Prod<N...>::value * D * N1;
    static constexpr long nbatch  = _CountInt<N...>::value + 2;

    template <long COUNT, bool dummy = true>
    struct returned { using type = typename SubArrayType::template returned<COUNT-1>::type; };
    template <bool dummy>
    struct returned<nbatch-1, dummy> { using type = PointType; };
    template <bool dummy>
    struct returned<0, dummy> { using type = SubArrayType; };

    virtual ~ConstStridedPointArray() {}

    template <typename Stride>
    __host__ __device__ 
    ConstStridedPointArray(const scalar_t * data, const Stride & stride): 
        data(data), stride(stride) {}

    __host__ __device__ 
    ConstStridedPointArray(scalar_t * data = nullptr): 
        data(data), stride() { stride.copy_(1); }

    template <typename... T>
    __host__ __device__  inline 
    const typename returned<_Count<T...>::value>::type at(offset_type n0, T... n) const
    { 
        return (*this)[n0].at(n...);
    };
    __host__ __device__ inline 
    const SubArrayType at(offset_type n0) const
    { 
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    __host__ __device__ inline 
    const SubArrayType operator[] (offset_type n0) const
    { 
        return this->at(n0);
    };

    const scalar_t * data;
    StaticPoint<nbatch+1, offset_t> stride;
};

template <long D, typename scalar_t, typename offset_t>
struct ConstStridedPointArray<D, scalar_t, offset_t> {

    using this_type         = ConstStridedPointArray<D, scalar_t, offset_t>;
    using value_type        = scalar_t;
    using offset_type       = offset_t;

    using PointType = StridedPoint<D, scalar_t, offset_t>;
    using ConstPointType = ConstStridedPoint<D, scalar_t, offset_t>;

    virtual ~ConstStridedPointArray() {}
    template <typename Stride>
    __host__ __device__ 
    ConstStridedPointArray(const scalar_t * data, const Stride & stride): 
        data(data), stride(stride) {}

    __host__ __device__ 
    ConstStridedPointArray(scalar_t * data = nullptr): 
        data(data), stride() { stride.copy_(1); }


    __host__ __device__  inline ConstPointType at(offset_type n) const
    { 
        return ConstPointType(data + n*stride[0], stride[1]); 
    };
    __host__ __device__  inline ConstPointType operator[] (offset_type n) const
    { 
        return this->at(n);
    };

    const scalar_t * data;
    StaticPoint<2, offset_t> stride;
};


} // namespace distance_mesh
} // namespace jf

#endif // JF_DISTANCE_MESH_UTILS_H