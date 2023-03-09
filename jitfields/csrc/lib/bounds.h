#ifndef JF_BOUNDS
#define JF_BOUNDS
#include "cuda_switch.h"
#include "atomic.h"
#include "utils.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             INDEXING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace jf {
namespace bound {

enum class type : char {
  Zero,         // Zero outside of the FOV
  Replicate,    // Replicate last inbound value = clip coordinates
  DCT1,         // Symmetric w.r.t. center of the last inbound voxel
  DCT2,         // Symmetric w.r.t. edge of the last inbound voxel (= Neumann)
  DST1,         // Antisymmetric w.r.t. center of the last inbound voxel
  DST2,         // Antisymmetric w.r.t. edge of the last inbound voxel (= Dirichlet)
  DFT,          // Circular / Wrap around the FOV
  NoCheck       // /!\ Checks disabled: assume coordinates are inbound
};

//namespace _index {

// These function act on floating point coordinates and simply
// apply the periodicity and reflection conditions of each boundary.
//
// This means that some of them output coordinates outside of the
// array support [0, n-1]. Coordinates would typically be converted to
// integer (round/floor/ceil), then clamped between [0, n-1], before
// being used to index into an array.
template <typename offset_t, bool is_float = is_floating_point<offset_t>::value >
struct _index
{
    template <typename size_t>
    static inline __device__
    offset_t inbounds(offset_t coord, size_t size)
    {
      return coord;
    }

    // Periodic (0, N-1)*2 + Reflect (0, N-1)
    // Support length = N-1
    // -> Boundary condition of a DCT-I
    template <typename size_t>
    static inline __device__
    offset_t reflect_Nminus1(offset_t coord, size_t size)
    {
      if (size == 1) return static_cast<offset_t>(0);
      size -= 1;
      size_t size_twice = size*2;
      coord = mod(abs(coord) % size_twice);                // period
      coord = coord > size ? size_twice - coord : coord;   // reflect
      return coord;
    }

    // Periodic (-1, N)*2 + Reflect (1, N)
    // Support length = N+1
    // -> Boundary condition of a DST-I
    template <typename size_t>
    static inline __device__
    offset_t reflect_Nplus1(offset_t coord, size_t size)
    {
      if (size == 1) static_cast<offset_t>(0);
      size += 1;
      size_t size_twice = size*2;
      coord += 1;
      coord = mod(abs(coord) % size_twice);                // period
      coord = coord > size ? size_twice - coord : coord;   // reflect
      coord -= 1;
      return coord;
    }

    // Periodic (-1/2, N-1/2)*2 + Reflect (-1/2, N-1/2)
    // Support length = N
    // -> Boundary condition of a DCT-II or DST-II
    template <typename size_t>
    static inline __device__
    offset_t reflect_N(offset_t coord, size_t size)
    {
      if (size == 1) static_cast<offset_t>(0);
      size_t size_twice = size*2;
      coord += 0.5;
      coord = mod(abs(coord) % size_twice);                // period
      coord = coord > size ? size_twice - coord : coord;   // reflect
      coord -= 0.5;
      return coord;
    }

    // Periodic (-1/2, N-1/2)
    // Support length = N
    // -> Boundary condition of a DFT
    template <typename size_t>
    static inline __device__
    offset_t circular(offset_t coord, size_t size)
    {
      if (size == 1) static_cast<offset_t>(0);
      coord += 0.5;
      coord = mod(coord % size);
      coord -= 0.5;
      return coord;
    }

    // Clamped to (-1/2, N-1/2)
    // Support length = N
    template <typename size_t>
    static inline __device__
    offset_t replicate(offset_t coord, size_t size)
    {
      coord = coord <= -0.5     ? static_cast<offset_t>(-0.5)
            : coord >= size-0.5 ? static_cast<offset_t>(size - 0.5) : coord;
      return coord;
    }
};

// These functions are specialized for integral coordinates
template <typename offset_t>
struct _index<offset_t, false>
{
    template <typename size_t>
    static inline __device__
    offset_t inbounds(offset_t coord, size_t size)
    {
      return coord;
    }

    // Boundary condition of a DCT-I (periodicity: (n-1)*2)
    // Indices are reflected about the centre of the border elements:
    //    -1 --> 1
    //     n --> n-2
    template <typename size_t>
    static inline __device__
    offset_t reflect_Nminus1(offset_t coord, size_t size)
    {
      if (size == 1) return 0;
      size_t size_twice = (size-1)*2;
      coord = abs(coord);
      coord = coord % size_twice;
      coord = coord >= size ? size_twice - coord : coord;
      return coord;
    }

    // Boundary condition of a DST-I (periodicity: (n+1)*2)
    // Indices are reflected about the centre of the first out-of-bound
    // element:
    //    -1 --> undefined [0]
    //    -2 --> 0
    //     n --> undefined [n-1]
    //   n+1 --> n-1
    template <typename size_t>
    static inline __device__
    offset_t reflect_Nplus1(offset_t coord, size_t size)
    {
      if (size == 1) return static_cast<offset_t>(0);
      size_t size_twice = (size+1)*2;
      coord = coord == -1 ? static_cast<offset_t>(0) : coord < 0 ? -coord-2 : coord;
      coord = coord % size_twice;
      coord = coord == size ? static_cast<offset_t>(size-1)
            : coord > size  ? size_twice-coord-2 : coord;
      return coord;
    }

    // Boundary condition of a DCT/DST-II (periodicity: n*2)
    // Indices are reflected about the edge of the border elements:
    //    -1 --> 0
    //     n --> n-1
    template <typename size_t>
    static inline __device__
    offset_t reflect_N(offset_t coord, size_t size)
    {
      size_t size_twice = size*2;
      coord = coord < 0 ? size_twice - ((-coord-1) % size_twice) - 1
                        : coord % size_twice;
      coord = coord >= size ? size_twice - coord - 1 : coord;
      return coord;
    }

    // Boundary condition of a DFT (periodicity: n)
    // Indices wrap about the edges:
    //    -1 --> n-1
    //     n --> 0
    template <typename size_t>
    static inline __device__
    offset_t circular(offset_t coord, size_t size)
    {
      coord = coord < 0 ? (size + coord%size) % size : coord % size;
      return coord;
    }

    // Replicate edge values:
    //    -1 --> 0
    //    -2 --> 0
    //     n --> n-1
    //   n+1 --> n-1
    template <typename size_t>
    static inline __device__
    offset_t replicate(offset_t coord, size_t size)
    {
      coord = coord <= 0    ? static_cast<offset_t>(0)
            : coord >= size ? static_cast<offset_t>(size - 1) : coord;
      return coord;
    }
};

//} // namespace index


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          SIGN MODIFICATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace _sign {

template <typename offset_t, typename size_t>
inline __device__ signed char inbounds(offset_t coord, size_t size) {
  return coord < 0 || coord >= size ? 0 : 1;
}

// Boundary condition of a DCT/DFT
// No sign modification based on coordinates
template <typename offset_t, typename size_t>
constexpr inline __device__ signed char constant(offset_t coord, size_t size) {
  return static_cast<signed char>(1);
}

// Boundary condition of a DST-I
// Periodic sign change based on coordinates
template <typename offset_t, typename size_t>
inline __device__ signed char periodic1(offset_t coord, size_t size) {
  if (size == 1) return 1;
  size_t size_twice = (size+1)*2;
  coord = coord < 0 ? size - coord - 1 : coord;
  coord = coord % size_twice;
  if (coord % (size+1) == size)   return  static_cast<signed char>(0);
  else if ((coord/(size+1)) % 2)  return  static_cast<signed char>(-1);
  else                            return  static_cast<signed char>(1);
}

// Boundary condition of a DST-II
// Periodic sign change based on coordinates
template <typename offset_t, typename size_t>
inline __device__ signed char periodic2(offset_t coord, size_t size) {
  coord = (coord < 0 ? size - coord - 1 : coord);
  return static_cast<signed char>((coord/size) % 2 ? -1 : 1);
}

} // namespace sign

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                BOUND
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check if coordinates within bounds
template <typename size_t>
inline __device__
bool inbounds(size_t coord, size_t size)
{
  return coord >= 0 && coord < size;
}

template <typename scalar_t, typename size_t>
inline __device__
bool inbounds(scalar_t coord, size_t size, scalar_t tol)
{
  return coord >= -tol && coord < (scalar_t)(size-1)+tol;
}

template <typename scalar_t, typename offset_t>
inline __device__
scalar_t get(const scalar_t * ptr, offset_t offset, signed char sign)
{
  if (sign == -1)  return -ptr[offset];
  else if (sign)   return  ptr[offset];
  else             return  static_cast<scalar_t>(0);
}

template <typename scalar_t, typename offset_t>
inline __device__
scalar_t get(const scalar_t * ptr, offset_t offset)
{
  return ptr[offset];
}

template <typename val_t, typename scalar_t, typename offset_t>
inline __device__
scalar_t cget(const scalar_t * ptr, offset_t offset, signed char sign)
{
  return static_cast<val_t>(get(ptr, offset, sign));
}

template <typename val_t, typename scalar_t, typename offset_t>
inline __device__
scalar_t cget(const scalar_t * ptr, offset_t offset)
{
  return static_cast<val_t>(get(ptr, offset));
}

template <typename scalar_t, typename offset_t, typename val_t>
inline __device__
void add(scalar_t *ptr, offset_t offset, val_t val, signed char sign)
{
  scalar_t cval = static_cast<scalar_t>(val);
  if (sign == -1)  anyAtomicAdd(ptr + offset, -cval);
  else if (sign)   anyAtomicAdd(ptr + offset,  cval);
}

template <typename scalar_t, typename offset_t, typename val_t>
inline __device__
void add(scalar_t *ptr, offset_t offset, val_t val)
{
  anyAtomicAdd(ptr + offset,  static_cast<scalar_t>(val));
}

template <type B> struct utils {
    template <typename offset_t, typename size_t>
    static inline __device__ offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::inbounds(coord, size); }
    template <typename offset_t, typename size_t>
    static inline __device__ signed char sign(offset_t coord, size_t size)
    { return _sign::inbounds(coord, size); }
};

template <> struct utils<type::Replicate> {
    template <typename offset_t, typename size_t>
    static inline __device__ offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::replicate(coord, size); }
    template <typename offset_t, typename size_t>
    static constexpr inline __device__ signed char sign(offset_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DCT1> {
    template <typename offset_t, typename size_t>
    static inline __device__ offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::reflect_Nminus1(coord, size); }
    template <typename offset_t, typename size_t>
    static constexpr inline __device__ signed char sign(offset_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DCT2> {
    template <typename offset_t, typename size_t>
    static inline __device__ offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::reflect_N(coord, size); }
    template <typename offset_t, typename size_t>
    static constexpr inline __device__ signed char sign(offset_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DST1> {
    template <typename offset_t, typename size_t>
    static inline __device__ offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::reflect_Nplus1(coord, size); }
    template <typename offset_t, typename size_t>
    static inline __device__ signed char sign(offset_t coord, size_t size)
    { return _sign::periodic1(coord, size); }
};

template <> struct utils<type::DST2> {
    template <typename offset_t, typename size_t>
    static inline __device__ offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::reflect_N(coord, size); }
    template <typename offset_t, typename size_t>
    static inline __device__ signed char sign(offset_t coord, size_t size)
    { return _sign::periodic2(coord, size); }
};

template <> struct utils<type::DFT> {
    template <typename offset_t, typename size_t>
    static inline __device__ offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::circular(coord, size); }
    template <typename offset_t, typename size_t>
    static constexpr inline __device__ signed char sign(offset_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

// Not iso -> use sign
template <type... B> struct getutils {
    template <typename val_t, typename scalar_t, typename offset_t>
    static inline __device__ scalar_t
    cget(const scalar_t * ptr, offset_t offset, signed char sign)
    { return cget<val_t>(ptr, offset, sign); }
    template <typename scalar_t, typename offset_t, typename val_t>
    static inline __device__ void
    add(scalar_t *ptr, offset_t offset, val_t val, signed char sign)
    { return add(ptr, offset, val, sign); }
};

// iso -> no need for sign

template <type B> struct getutils<B> {
    template <typename val_t, typename scalar_t, typename offset_t>
    static inline __device__ scalar_t
    cget(const scalar_t * ptr, offset_t offset, signed char)
    { return bound::cget<val_t>(ptr, offset); }
    template <typename scalar_t, typename offset_t, typename val_t>
    static inline __device__ void
    add(scalar_t *ptr, offset_t offset, val_t val, signed char)
    { return bound::add(ptr, offset, val); }
};

template <type B> struct getutils<B,B> {
    template <typename val_t, typename scalar_t, typename offset_t>
    static inline __device__ scalar_t
    cget(const scalar_t * ptr, offset_t offset, signed char)
    { return bound::cget<val_t>(ptr, offset); }
    template <typename scalar_t, typename offset_t, typename val_t>
    static inline __device__ void
    add(scalar_t *ptr, offset_t offset, val_t val, signed char)
    { return bound::add(ptr, offset, val); }
};

template <type B> struct getutils<B,B,B> {
    template <typename val_t, typename scalar_t, typename offset_t>
    static inline __device__ scalar_t
    cget(const scalar_t * ptr, offset_t offset, signed char)
    { return bound::cget<val_t>(ptr, offset); }
    template <typename scalar_t, typename offset_t, typename val_t>
    static inline __device__ void
    add(scalar_t *ptr, offset_t offset, val_t val, signed char)
    { return bound::add(ptr, offset, val); }
};

// unless dst/zero

#define JF_ISO_SIGN(B) \
    template <> struct getutils<B> { \
        template <typename val_t, typename scalar_t, typename offset_t> \
        static inline __device__ scalar_t \
        cget(const scalar_t * ptr, offset_t offset, signed char sign) \
        { return bound::cget<val_t>(ptr, offset, sign); } \
        template <typename scalar_t, typename offset_t, typename val_t> \
        static inline __device__ void \
        add(scalar_t *ptr, offset_t offset, val_t val, signed char sign) \
        { return bound::add(ptr, offset, val, sign); } \
    }; \
    template <> struct getutils<B,B> { \
        template <typename val_t, typename scalar_t, typename offset_t> \
        static inline __device__ scalar_t \
        cget(const scalar_t * ptr, offset_t offset, signed char sign) \
        { return bound::cget<val_t>(ptr, offset, sign); } \
        template <typename scalar_t, typename offset_t, typename val_t> \
        static inline __device__ void \
        add(scalar_t *ptr, offset_t offset, val_t val, signed char sign) \
        { return bound::add(ptr, offset, val, sign); } \
    }; \
    template <> struct getutils<B,B,B> { \
        template <typename val_t, typename scalar_t, typename offset_t> \
        static inline __device__ scalar_t \
        cget(const scalar_t * ptr, offset_t offset, signed char sign) \
        { return bound::cget<val_t>(ptr, offset, sign); } \
        template <typename scalar_t, typename offset_t, typename val_t> \
        static inline __device__ void \
        add(scalar_t *ptr, offset_t offset, val_t val, signed char sign) \
        { return bound::add(ptr, offset, val, sign); } \
    };

JF_ISO_SIGN(type::DST1)
JF_ISO_SIGN(type::DST2)
JF_ISO_SIGN(type::Zero)

template <typename offset_t, typename size_t>
static inline __device__ offset_t index(type bound_type, offset_t coord, size_t size) {
  switch (bound_type) {
    case type::Replicate:  return _index<offset_t>::replicate(coord, size);
    case type::DCT1:       return _index<offset_t>::reflect_Nminus1(coord, size);
    case type::DCT2:       return _index<offset_t>::reflect_N(coord, size);
    case type::DST1:       return _index<offset_t>::reflect_Nplus1(coord, size);
    case type::DST2:       return _index<offset_t>::reflect_N(coord, size);
    case type::DFT:        return _index<offset_t>::circular(coord, size);
    case type::Zero:       return _index<offset_t>::inbounds(coord, size);
    default:               return _index<offset_t>::inbounds(coord, size);
  }
}

template <typename offset_t, typename size_t>
static inline __device__ signed char sign(type bound_type, offset_t coord, size_t size) {
  switch (bound_type) {
    case type::Replicate:  return _sign::constant(coord, size);
    case type::DCT1:       return _sign::constant(coord, size);
    case type::DCT2:       return _sign::constant(coord, size);
    case type::DST1:       return _sign::periodic1(coord, size);
    case type::DST2:       return _sign::periodic2(coord, size);
    case type::DFT:        return _sign::constant(coord, size);
    case type::Zero:       return _sign::inbounds(coord, size);
    default:               return _sign::inbounds(coord, size);
  }
}

} // namespace bound
} // namespace jf

#endif // JF_BOUNDS
