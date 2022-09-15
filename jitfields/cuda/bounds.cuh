/* DEPENDENCIES:
 * #include "atomic.cuh"
 */

#ifndef JF_BOUNDS
#define JF_BOUNDS

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             INDEXING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace bound {

enum class type : char {
  Zero,         // Zero outside of the FOV
  Replicate,    // Replicate last inbound value = clip coordinates
  DCT1,         // Symetric w.r.t. center of the last inbound voxel
  DCT2,         // Symetric w.r.t. edge of the last inbound voxel (=Neuman)
  DST1,         // Antisymetric w.r.t. center of the last inbound voxel
  DST2,         // Antisymetric w.r.t. edge of the last inbound voxel (=Dirichlet)
  DFT,          // Circular / Wrap arounf the FOV
  NoCheck       // /!\ Checks disabled: assume coordinates are inbound
};

namespace _index {

template <typename size_t>
static inline __device__ size_t inbounds(size_t coord, size_t size) {
  return coord;
}

// Boundary condition of a DCT-I (periodicity: (n-1)*2)
// Indices are reflected about the centre of the border elements:
//    -1 --> 1
//     n --> n-2
template <typename size_t>
static inline __device__ size_t reflect1c(size_t coord, size_t size) {
  if (size == 1) return 0;
  size_t size_twice = (size-1)*2;
  coord = coord < 0 ? -coord : coord;
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
static inline __device__ size_t reflect1s(size_t coord, size_t size) {
  if (size == 1) return 0;
  size_t size_twice = (size+1)*2;
  coord = coord == -1 ? 0 : coord < 0 ? -coord-2 : coord;
  coord = coord % size_twice;
  coord = coord == size ? size-1 : coord > size ? size_twice-coord-2 : coord;
  return coord;
}

// Boundary condition of a DCT/DST-II (periodicity: n*2)
// Indices are reflected about the edge of the border elements:
//    -1 --> 0
//     n --> n-1
template <typename size_t>
static inline __device__ size_t reflect2(size_t coord, size_t size) {
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
static inline __device__ size_t circular(size_t coord, size_t size) {
  coord = coord < 0 ? (size + coord%size) % size : coord % size;
  return coord;
}

// Replicate edge values:
//    -1 --> 0
//    -2 --> 0
//     n --> n-1
//   n+1 --> n-1
template <typename size_t>
static inline __device__ size_t replicate(size_t coord, size_t size) {
  coord = coord <= 0 ? 0 : coord >= size ? size - 1 : coord;
  return coord;
}

} // namespace index


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          SIGN MODIFICATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace _sign {

template <typename size_t>
static inline __device__ signed char inbounds(size_t coord, size_t size) {
  return coord < 0 || coord >= size ? 0 : 1;
}

// Boundary condition of a DCT/DFT
// No sign modification based on coordinates
template <typename size_t>
static inline __device__ signed char constant(size_t coord, size_t size) {
  return static_cast<signed char>(1);
}

// Boundary condition of a DST-I
// Periodic sign change based on coordinates
template <typename size_t>
static inline __device__ signed char periodic1(size_t coord, size_t size) {
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
template <typename size_t>
static inline __device__ signed char periodic2(size_t coord, size_t size) {
  coord = (coord < 0 ? size - coord - 1 : coord);
  return static_cast<signed char>((coord/size) % 2 ? -1 : 1);
}

} // namespace sign

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                BOUND
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check if coordinates within bounds
template <typename size_t>
static inline __device__ bool inbounds(size_t coord, size_t size) {
  return coord >= 0 && coord < size;
}

template <typename scalar_t, typename size_t>
static inline __device__ bool inbounds(scalar_t coord, size_t size, scalar_t tol) {
  return coord >= -tol && coord < (scalar_t)(size-1)+tol;
}

template <typename scalar_t, typename offset_t>
static inline __device__ scalar_t
get(const scalar_t * ptr, offset_t offset,
    signed char sign = static_cast<signed char>(1)) {
  if (sign == -1)  return -ptr[offset];
  else if (sign)   return  ptr[offset];
  else             return  static_cast<scalar_t>(0);
}

template <typename scalar_t, typename offset_t>
static inline __device__ void
add(scalar_t *ptr, offset_t offset, scalar_t val,
    signed char sign = static_cast<signed char>(1)) {
  if (sign == -1)  gpuAtomicAdd(ptr + offset, -val);
  else if (sign)   gpuAtomicAdd(ptr + offset,  val);
}

template <type B> struct utils {
    template <typename size_t>
    static inline __device__ size_t index(size_t coord, size_t size)
    { return _index::inbounds(coord, size); }
    template <typename size_t>
    static inline __device__ signed char sign(size_t coord, size_t size)
    { return _sign::inbounds(coord, size); }
};

template <> struct utils<type::Replicate> {
    template <typename size_t>
    static inline __device__ size_t index(size_t coord, size_t size)
    { return _index::replicate(coord, size); }
    template <typename size_t>
    static inline __device__ signed char sign(size_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DCT1> {
    template <typename size_t>
    static inline __device__ size_t index(size_t coord, size_t size)
    { return _index::reflect1c(coord, size); }
    template <typename size_t>
    static inline __device__ signed char sign(size_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DCT2> {
    template <typename size_t>
    static inline __device__ size_t index(size_t coord, size_t size)
    { return _index::reflect2(coord, size); }
    template <typename size_t>
    static inline __device__ signed char sign(size_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DST1> {
    template <typename size_t>
    static inline __device__ size_t index(size_t coord, size_t size)
    { return _index::reflect1s(coord, size); }
    template <typename size_t>
    static inline __device__ signed char sign(size_t coord, size_t size)
    { return _sign::periodic1(coord, size); }
};

template <> struct utils<type::DST2> {
    template <typename size_t>
    static inline __device__ size_t index(size_t coord, size_t size)
    { return _index::reflect2(coord, size); }
    template <typename size_t>
    static inline __device__ signed char sign(size_t coord, size_t size)
    { return _sign::periodic2(coord, size); }
};

template <> struct utils<type::DFT> {
    template <typename size_t>
    static inline __device__ size_t index(size_t coord, size_t size)
    { return _index::circular(coord, size); }
    template <typename size_t>
    static inline __device__ signed char sign(size_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <typename size_t>
static inline __device__ size_t index(type bound_type, size_t coord, size_t size) {
  switch (bound_type) {
    case type::Replicate:  return _index::replicate(coord, size);
    case type::DCT1:       return _index::reflect1c(coord, size);
    case type::DCT2:       return _index::reflect2(coord, size);
    case type::DST1:       return _index::reflect1s(coord, size);
    case type::DST2:       return _index::reflect2(coord, size);
    case type::DFT:        return _index::circular(coord, size);
    case type::Zero:       return _index::inbounds(coord, size);
    default:               return _index::inbounds(coord, size);
  }
}

template <typename size_t>
static inline __device__ signed char sign(type bound_type, size_t coord, size_t size) {
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

#endif // JF_BOUNDS