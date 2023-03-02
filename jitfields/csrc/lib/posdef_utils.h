#ifndef JF_POSDEF_UTILS
#define JF_POSDEF_UTILS
#include "cuda_switch.h"
#include "utils.h"

#define JFH_OnePlusTiny 1.000001
#define JF_UNUSED __attribute__((unused))

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              INTERNALS
//
// ---------------------------------------------------------------------
//
// We first define a bunch of internal utilities that allow:
// 1) to work with strided pointers. Strided pointers point to memory
//    in which elements of interest are not separated by `sizeof(T)`
//    but by `S * sizeof(T)`. This classes implement operators that are
//    classically used on pointers (dereference, access, ++, +=, --, -=).
// 2) we define traits that work on both classical and strided pointers:
//      elem_type<T>::value -> Type of referenced elements
//      is_const<T>::value -> Whether referenced elements are const
//      return_type<T...>::value -> Upcast of types T...
// 3) we define inplace operators that convert all values to a "reduction"
//    type to carry the computation before downcasting to the output type.
//    E.g.: iadd<reduce_t>, isub<reduce_t>, iaddcmul<reduce_t>, ...
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


namespace jf {
namespace posdef {
namespace internal {

//----------------------------------------------------------------------
//
//                          Strided pointers
//
//    We define a class Pointer<T, S, L>  in which the stride
//    length is statically known, and a specialized version Pointer<T, 0>
//    for strides that are only known dynamically.
//
//----------------------------------------------------------------------

template <typename ST, long S=1, typename OT=long>
struct Pointer {
    using this_type = Pointer<ST, S, OT>;
    using scalar_t = ST;
    using offset_t = OT;

    scalar_t * data;
    static constexpr offset_t stride = static_cast<offset_t>(S);

    Pointer(scalar_t * ptr): data(ptr) {}
    Pointer(const Pointer<scalar_t, S, offset_t> & ptr): data(ptr.data) {}

    inline __device__ scalar_t& operator[] (offset_t i) const { return data[i*stride]; }
    // inline __device__ const scalar_t& operator[] (offset_t i) const { return data[i*stride]; }
    inline __device__ scalar_t& operator* () const { return *data; }
    inline __device__ operator bool () const { return data != nullptr; }

    inline __device__ this_type & operator++ () { data += stride; return *this; }
    inline __device__ this_type operator++ (int) { this_type prev = *this; data += stride; return prev; }
    inline __device__ this_type & operator-- () { data -= stride; return *this; }
    inline __device__ this_type operator-- (int) { this_type prev = *this; data -= stride; return prev; }
    inline __device__ this_type & operator += (offset_t N) { data += N * stride; return *this; }
    inline __device__ this_type & operator -= (offset_t N) { data -= N * stride; return *this; }
};

template <typename ST, typename OT>
struct Pointer<ST, 0, OT> {
    using this_type = Pointer<ST, 0, OT>;
    using scalar_t = ST;
    using offset_t = OT;

    scalar_t * data;
    offset_t stride;

    Pointer(scalar_t * ptr): data(ptr), stride(1) {}
    Pointer(scalar_t * ptr, offset_t str): data(ptr), stride(str) {}

    template <typename inp_offset_t, long S>
    Pointer(const Pointer<scalar_t, S, inp_offset_t> & ptr):
        data(ptr.data), stride(static_cast<offset_t>(ptr.stride)) {}

    inline __device__ scalar_t& operator[] (offset_t i) const { return data[i*stride]; }
    // inline __device__ const scalar_t& operator[] (offset_t i) const { return data[i*stride]; }
    inline __device__ scalar_t& operator* () const { return *data; }
    inline __device__ operator bool () const { return data != nullptr; }

    inline __device__ this_type & operator++ () { data += stride; return *this; }
    inline __device__ this_type operator++ (int) { this_type prev = *this; data += stride; return prev; }
    inline __device__ this_type & operator-- () { data -= stride; return *this; }
    inline __device__ this_type operator-- (int) { this_type prev = *this; data -= stride; return prev; }
    inline __device__ this_type & operator += (offset_t N) { data += N * stride; return *this; }
    inline __device__ this_type & operator -= (offset_t N) { data -= N * stride; return *this; }
};

template <typename ST, typename OT>
using StridedPointer = Pointer<ST, 0, OT>;

#if 0
template <typename scalar_t, long S, typename offset_t>
std::ostream& operator<< (std::ostream& os, const Pointer<scalar_t, S, offset_t> & ptr)
{
    os << "Pointer[" << ptr.data << " (" << ptr.stride << ")]";
    return os;
}
#endif

template <typename scalar_t, long S, typename offset_t>
inline __device__
Pointer<scalar_t, S, offset_t> operator+ (Pointer<scalar_t, S, offset_t> prev, offset_t N)
{
    Pointer<scalar_t, S, offset_t> next = prev;
    next += N;
    return next;
}

template <typename scalar_t, long S, typename offset_t>
inline __device__
Pointer<scalar_t, S, offset_t> operator- (Pointer<scalar_t, S, offset_t> prev, offset_t N)
{
    Pointer<scalar_t, S, offset_t> next = prev;
    next -= N;
    return next;
}

template <typename scalar_t, long S, typename offset_t>
inline __device__
Pointer<scalar_t, S, offset_t> pointer(Pointer<scalar_t, S, offset_t> ptr)
{
    return ptr;
}

template <typename scalar_t, typename offset_t>
inline __device__
Pointer<scalar_t, 0, offset_t> pointer(scalar_t * ptr, offset_t stride)
{
    return Pointer<scalar_t, 0, offset_t>(ptr, stride);
}

template <typename scalar_t>
inline __device__
Pointer<scalar_t, 0, long> pointer(scalar_t * ptr)
{
    return Pointer<scalar_t, 0, long>(ptr);
}


//----------------------------------------------------------------------
//
//                              Traits
//
//----------------------------------------------------------------------

// ----------------
// traits: is_pointer, as_pointer
// Convert classic pointer types into our Pointer type
// ----------------

template <typename T>
struct as_pointer {};

template <typename T>
struct is_pointer { static constexpr bool value = false; };

template <typename scalar_t, long S, typename offset_t>
struct as_pointer<Pointer<scalar_t, S, offset_t> > {
    using value = Pointer<scalar_t, S, offset_t>;
};

template <typename scalar_t, long S, typename offset_t>
struct is_pointer<Pointer<scalar_t, S, offset_t> > {
    static constexpr bool value = true;
};

template <typename scalar_t>
struct as_pointer<scalar_t *> {
    using value = Pointer<scalar_t>;
};

template <typename scalar_t>
struct is_pointer<scalar_t *> {
    static constexpr bool value = true;
};

// ----------------
// traits: deconst
// Remove constness from scalar type
// ----------------

template <typename T>
struct deconst {
    using value = T;
};

template <typename T>
struct deconst<const T> {
    using value = T;
};

// ----------------
// traits: elem_type
// Return the element-type (without constness) referenced by a pointer
// ----------------

template <typename T, bool is_pointer_type = is_pointer<T>::value>
struct elem_type {
    using value = typename deconst<typename as_pointer<T>::value::scalar_t>::value;
};

template <typename T>
struct elem_type<T, false> {
    using value = typename deconst<T>::value;
};

// ----------------
// traits: upcast
// Return the output type of a binary (or +) operation on two (or +) types
// ----------------

// -- Helper for dealing with a single type

template <typename left_t>
struct return_type1 {
    using value = typename elem_type<left_t>::value;
};

// -- Helper for dealing with a pair of types

template <typename left_t, typename right_t>
struct return_type2 {
    using value = typename return_type2<
        typename return_type1<left_t>::value,
        typename return_type1<right_t>::value>::value;
};

template <typename same_t>
struct return_type2<same_t, same_t> {
    using value = typename return_type1<same_t>::value;
};

// void gets skipped
template <typename scalar_t>
struct return_type2<scalar_t, void> {
    using value = typename return_type1<scalar_t>::value;
};
template <typename scalar_t>
struct return_type2<void, scalar_t> {
    using value = typename return_type1<scalar_t>::value;
};


// <float, double> -> double
template <>
struct return_type2<float, double> {
    using value = double;
};
template <>
struct return_type2<double, float> {
    using value = double;
};

#ifdef __CUDACC__
    // <half, double> -> double
    template <>
    struct return_type2<double, half> {
        using value = double;
    };
    template <>
    struct return_type2<half, double> {
        using value = double;
    };

    // <half, float> -> float
    template <>
    struct return_type2<float, half> {
        using value = float;
    };
    template <>
    struct return_type2<half, float> {
        using value = float;
    };

#endif // __CUDACC__


// -- Generic declaration

// Should never be called unless there is only one type
// Then, we fallback to the first type.
template <typename left_t, typename... scalar_t>
struct return_type {
    using value = return_type1<left_t>;
};

// 2 types -> defer to return_type2
template <typename left_t, typename right_t>
struct return_type<left_t, right_t> {
    using value = return_type2<left_t, right_t>;
};

// 3+ types -> collapse first two types and recurse
template <typename left_t, typename right_t, typename next_t, typename... other_t>
struct return_type<left_t, right_t, next_t, other_t...> {
    using _lr = typename return_type2<left_t, right_t>::value;
    using value = typename return_type<_lr, next_t, other_t...>::value;
};

//----------------------------------------------------------------------
//
//        In-place operators with upcast to reduction type
//
//----------------------------------------------------------------------

// left = right
template <typename left_t, typename right_t>
inline __device__ void set(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(right);
}

// left += right
template <typename reduce_t, typename left_t, typename right_t>
inline __device__ void iadd(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(static_cast<reduce_t>(left) +
                               static_cast<reduce_t>(right));
}

// left -= right
template <typename reduce_t, typename left_t, typename right_t>
inline __device__ void isub(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(static_cast<reduce_t>(left) -
                               static_cast<reduce_t>(right));
}

// left *= right
template <typename reduce_t, typename left_t, typename right_t>
inline __device__ void imul(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(static_cast<reduce_t>(left) *
                               static_cast<reduce_t>(right));
}

// left /= right
template <typename reduce_t, typename left_t, typename right_t>
inline __device__ void idiv(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(static_cast<reduce_t>(left) /
                               static_cast<reduce_t>(right));
}

// out += left * right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline __device__ void iaddcmul(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(out) +
                               static_cast<reduce_t>(left) *
                               static_cast<reduce_t>(right));
}

// out -= left * right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline __device__ void isubcmul(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(out) -
                               static_cast<reduce_t>(left) *
                               static_cast<reduce_t>(right));
}

// out /= left + right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline __device__ void idivcadd(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(out) /
                               (static_cast<reduce_t>(left) +
                                static_cast<reduce_t>(right)));
}

// out = left + right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline __device__ void add(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(left) +
                             static_cast<reduce_t>(right));
}

// out = left - right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline __device__ void sub(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(left) -
                             static_cast<reduce_t>(right));
}

// out = left * right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline __device__ void mul(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(left) *
                             static_cast<reduce_t>(right));
}

// out = left / right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline __device__ void div(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(left) /
                             static_cast<reduce_t>(right));
}

} // namespace internal
} // namespace posdef
} // namespace jf

#endif // JF_POSDEF_UTILS
