#ifndef JF_CUDA_SWITCH
#define JF_CUDA_SWITCH

#ifndef __CUDA__
// replace __device__ with empty symbol
#define __device__
#else
#include <cuda_fp16.h>
#endif

#endif // JF_CUDA_SWITCH
