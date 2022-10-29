#ifndef JF_PARALLEL_H
#define JF_PARALLEL_H
#include "threadpool.h"
#include "parallel_impl.h"

/* LICENSE:
 * Most of the functions are adapted from PyTorch/ATen's ParallelNative
 * https://github.com/pytorch/pytorch/blob/master/LICENSE
 */

namespace jf {

constexpr long GRAIN_SIZE = 32768;

template <class F>
inline void parallel_for(long begin, long end, long grain_size, const F& f)
{
    if (begin >= end) return;

    const auto numiter = end - begin;
    const bool use_parallel =  (numiter > grain_size && numiter > 1 &&
                                // !internal::in_parallel_region() &&
                                get_num_threads() > 1);
    if (!use_parallel) {
        // internal::ThreadIdGuard tid_guard(0);
        f(begin, end);
        return;
    }

    internal::invoke_parallel(begin, end, grain_size, f);
}

} // namespace jf

#endif // JF_PARALLEL_H
