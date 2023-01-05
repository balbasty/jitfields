#ifndef JF_PARALLEL_H
#define JF_PARALLEL_H

/* While <future> (and therefore our thread pool) works fine on MacOS,
 * it fails on Linux, apparently because of the older LLVM under the hood.
 * See:
 *      https://github.com/wlav/cppyy/issues/60
 *      https://stackoverflow.com/questions/73424050
 * For now I am checking the llvm (clang) version and only enable the
 * thread pool if is is recent enough. My mac laptop has version 14, and
 * one of the issues above mentions version 13, so I enable it for
 * versions >= 13.
 *
 * Note that I had to disable the use of thread_local in parallel_impl.h,
 * which fail even on MacOS.
 */
#define JF_CAN_USE_PARALLEL 1
#if __clang__
#if __clang_major__ < 13
#undef  JF_CAN_USE_PARALLEL
#define JF_CAN_USE_PARALLEL 0
#endif
#endif

#if JF_CAN_USE_PARALLEL
#  include "threadpool.h"
#  include "parallel_impl.h"  // cannot import until cppyy is fixed
#endif

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

#if !(JF_CAN_USE_PARALLEL)
    f(begin, end);
    return;
#else
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
#endif
}

} // namespace jf

#endif // JF_PARALLEL_H
