/***********************************************************************
 * copied from PyTorch/ATen
 * https://github.com/pytorch/pytorch/blob/master/LICENSE
 **********************************************************************/

#ifndef JF_ATOMIC
#define JF_ATOMIC

template <typename T>
struct AtomicFPOp;

template <>
struct AtomicFPOp<double> {
  template <typename func_t>
  inline __device__ double operator() (double * address, double val, const func_t& func) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, func(val, assumed));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
  }
};

#define ATOMIC_INTEGER_IMPL(NAME)                                                                                      \
template <typename T, size_t n>                                                                                        \
struct Atomic##NAME##IntegerImpl;                                                                                      \
                                                                                                                       \
template<typename T>                                                                                                   \
struct Atomic##NAME##IntegerImpl<T, 1> {                                                                               \
  template <typename func_t>                                                                                           \
  inline __device__ void operator()(T *address, T val, const func_t& func) {                                           \
    size_t offset = (size_t)address & 3;                                                                               \
    unsigned int * address_as_ui = (unsigned int *)((char *)address - offset);                                         \
    unsigned int old = *address_as_ui;                                                                                 \
    unsigned int shift = offset * 8;                                                                                   \
    unsigned int old_byte;                                                                                             \
    unsigned int newval;                                                                                               \
    unsigned int assumed;                                                                                              \
                                                                                                                       \
    do {                                                                                                               \
      assumed = old;                                                                                                   \
      old_byte = (old >> shift) & 0xff;                                                                                \
      newval = static_cast<char>(func(val, static_cast<T>(old_byte)));                                                 \
      newval = (old & ~(0x000000ff << shift)) | (newval << shift);                                                     \
      old = atomicCAS(address_as_ui, assumed, newval);                                                                 \
    } while (assumed != old);                                                                                          \
  }                                                                                                                    \
};                                                                                                                     \
                                                                                                                       \
template<typename T>                                                                                                   \
struct Atomic##NAME##IntegerImpl<T, 2> {                                                                               \
  template <typename func_t>                                                                                           \
  inline __device__ void operator()(T *address, T val, const func_t& func) {                                           \
    size_t offset = (size_t)address & 2;                                                                               \
    unsigned int * address_as_ui = (unsigned int *)((char *)address - offset);                                         \
    bool is_32_align = offset;                                                                                         \
    unsigned int old = *address_as_ui;                                                                                 \
    unsigned int old_bytes;                                                                                            \
    unsigned int newval;                                                                                               \
    unsigned int assumed;                                                                                              \
                                                                                                                       \
    do {                                                                                                               \
      assumed = old;                                                                                                   \
      old_bytes = is_32_align ? old >> 16 : old & 0xffff;                                                              \
      newval = static_cast<unsigned short>(func(val, static_cast<T>(old_bytes)));                                      \
      newval = is_32_align ? (old & 0xffff) | (newval << 16) : (old & 0xffff0000) | newval;                            \
      old = atomicCAS(address_as_ui, assumed, newval);                                                                 \
    } while (assumed != old);                                                                                          \
  }                                                                                                                    \
};                                                                                                                     \
                                                                                                                       \
template<typename T>                                                                                                   \
struct Atomic##NAME##IntegerImpl<T, 4> {                                                                               \
  template <typename func_t>                                                                                           \
  inline __device__ void operator()(T *address, T val, const func_t& func) {                                           \
    unsigned int * address_as_ui = (unsigned int *) (address);                                                         \
    unsigned int old = *address_as_ui;                                                                                 \
    unsigned int newval;                                                                                               \
    unsigned int assumed;                                                                                              \
                                                                                                                       \
    do {                                                                                                               \
      assumed = old;                                                                                                   \
      newval = static_cast<unsigned int>(func(val, static_cast<T>(old)));                                              \
      old = atomicCAS(address_as_ui, assumed, newval);                                                                 \
    } while (assumed != old);                                                                                          \
  }                                                                                                                    \
};                                                                                                                     \
                                                                                                                       \
template<typename T>                                                                                                   \
struct Atomic##NAME##IntegerImpl<T, 8> {                                                                               \
  template <typename func_t>                                                                                           \
  inline __device__ void operator()(T *address, T val, const func_t& func) {                                           \
    unsigned long long * address_as_ui = (unsigned long long *) (address);                                             \
    unsigned long long old = *address_as_ui;                                                                           \
    unsigned long long newval;                                                                                         \
    unsigned long long assumed;                                                                                        \
                                                                                                                       \
    do {                                                                                                               \
      assumed = old;                                                                                                   \
      newval = static_cast<unsigned long>(func(val, static_cast<T>(old)));                                             \
      old = atomicCAS(address_as_ui, assumed, newval);                                                                 \
    } while (assumed != old);                                                                                          \
  }                                                                                                                    \
};


# define GPU_ATOMIC_INTEGER(NAME, OP, DTYPE)                                                                           \
static inline __device__ void gpuAtomic##NAME(DTYPE *address, DTYPE val) {                                             \
Atomic##NAME##IntegerImpl<DTYPE, sizeof(DTYPE)>()(address,                                                             \
                                                      val,                                                             \
                                                      [](DTYPE a, DTYPE b) {                                           \
                                                          return OP;                                                   \
                                                      });                                                              \
}                                                                                                                      \

ATOMIC_INTEGER_IMPL(Add)

/*
// Don't instantiate gpuAtomicAdd with the macro as it seems non-standard (see int32, int64)
static inline __device__ void gpuAtomicAdd(char *address, char val) {
  AtomicAddIntegerImpl<char, sizeof(char)>()(address,
                                                   val,
                                                   [](char a, char b) {
                                                      return a + b;
                                                   });
}

static inline  __device__ void gpuAtomicAdd(signed char *address, signed char val) {
  AtomicAddIntegerImpl<signed char, sizeof(signed char)>()(address,
                                                 val,
                                                 [](signed char a, signed char b) {
                                                   return a + b;
                                                 });
}

static inline  __device__ void gpuAtomicAdd(short *address, short val) {
  AtomicAddIntegerImpl<short, sizeof(short)>()(address,
                                                   val,
                                                   [](short a, short b) {
                                                     return a + b;
                                                   });
}

static inline __device__ int gpuAtomicAdd(int *address, int val) {
  return atomicAdd(address, val);
}

static inline __device__ void gpuAtomicAdd(long *address, long val) {
#if defined(USE_ROCM)
  __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#else
  AtomicAddIntegerImpl<long, sizeof(long)>()(address,
                                                   val,
                                                   [](long a, long b) {
                                                      return a + b;
                                                   });
#endif
}

static inline __device__ void gpuAtomicAdd(bool *address, bool val) {
  *address = address && val;
}
*/

// from CUDA C Programmic Guide
static inline __device__ double atomicAdd(double* address, double val)
#if defined(__clang__) && defined(__CUDA__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"
    __attribute__((enable_if(true, "")))
#pragma GCC diagnostic pop
#endif // defined(__clang__) && defined(__CUDA__)
{

  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(val + __longlong_as_double(assumed));
                              });
}

static inline __device__ double gpuAtomicAdd(double *address, double val) {
  return atomicAdd(address, val);
}

static inline __device__ float gpuAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

/*
template<typename T>
static inline __device__ void gpuAtomicAdd(complex<T> *address, complex<T> val) {
  gpuAtomicAdd(&address->real_, val.real_);
  gpuAtomicAdd(&address->imag_, val.imag_);
}
*/

/* Note [gpuAtomicAdd vs atomicAdd]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Some extensions such as torchvision call atomicAdd()
 * directly and require non-library provided data type support. Only for these, we
 * continue to provide atomicAdd overloads.
 */

/*
static inline __device__ void atomicAdd(char *address, char val) {
  gpuAtomicAdd(address, val);
}

static inline  __device__ void atomicAdd(signed char *address, signed char val) {
  gpuAtomicAdd(address, val);
}

static inline  __device__ void atomicAdd(short *address, short val) {
  gpuAtomicAdd(address, val);
}

static inline __device__ void atomicAdd(long *address, long val) {
  gpuAtomicAdd(address, val);
}

static inline __device__ void atomicAdd(bool *address, bool val) {
  gpuAtomicAdd(address, val);
}
*/

/* Note [explicitly non-returning atomics]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * AMD's MI100 (gfx908) provides an optimized fp32 atomicAdd, exposed via atomicAddNoRet().
 * Due to compiler limitations, callers must opt-in to guarantee the optimized instruction.
 * This non-returning atomicAddNoRet cannot be used to implement the returning atomicAdd,
 * therefore we need a new API 'gpuAtomicAddNoReturn'.
 */
/*
template<typename T>
static inline __device__ void gpuAtomicAddNoReturn(complex<T> *address, complex<T> val) { gpuAtomicAdd(address, val); }
static inline __device__ void gpuAtomicAddNoReturn(char *address, char val) { gpuAtomicAdd(address, val); }
static inline __device__ void gpuAtomicAddNoReturn(signed char *address, signed char val) { gpuAtomicAdd(address, val); }
static inline __device__ void gpuAtomicAddNoReturn(short *address, short val) { gpuAtomicAdd(address, val); }
static inline __device__ void gpuAtomicAddNoReturn(int *address, int val) { gpuAtomicAdd(address, val); }
static inline __device__ void gpuAtomicAddNoReturn(long *address, long val) { gpuAtomicAdd(address, val); }
static inline __device__ void gpuAtomicAddNoReturn(bool *address, bool val) { gpuAtomicAdd(address, val); }
*/
static inline __device__ void gpuAtomicAddNoReturn(double *address, double val) { gpuAtomicAdd(address, val); }

/* Special case fp32 atomic. */
#if defined(USE_ROCM)
static inline __device__ void gpuAtomicAddNoReturn(float *address, float val) { atomicAddNoRet(address, val); }
#else
static inline __device__ void gpuAtomicAddNoReturn(float *address, float val) { gpuAtomicAdd(address, val); }
#endif

#endif // JF_ATOMIC