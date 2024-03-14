#ifndef JF_CUDA_SWITCH
#define JF_CUDA_SWITCH

#ifndef __CUDACC__
// replace __device__ with empty symbol
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#else
#include <cuda_fp16.h>
#endif

#endif // JF_CUDA_SWITCH
