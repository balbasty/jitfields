/* LICENSE:
 * Most of the functions are adapted from PyTorch/ATen's ParallelNative
 * https://github.com/pytorch/pytorch/blob/master/LICENSE
 */
#ifndef JF_PARALLEL_IMPL_H
#define JF_PARALLEL_IMPL_H

/* While <future> (and therefore our thread pool) works fine on MacOS,
 * it fails on Linux, apparently because of the older LLVM under the hood.
 * See:
 *      https://github.com/wlav/cppyy/issues/60
 *      https://stackoverflow.com/questions/73424050
 * For now I am checking the llvm (clang) version and only enable the
 * thread pool if is is recent enough. My mac laptop has version 14, and
 * one of the issues above mentions version 13, so I enable it for
 * versions >= 13.
 * ---------------------------------------------------------------------
 * I have removed the in_parallel_region/get_thread_num API because
 * it requires the use of "thread_local" which does not work in cppyy
 * on MacOS.
 * I think it's fine for now because we always call parallel_for from
 * the main thread (so no use for in_parallel_region) and we don't use
 * the thread id in our loops (no use for get_thread_num).
 */

#ifdef JF_USE_SEQ
#   undef  JF_CAN_USE_FUTURE
#   define JF_CAN_USE_FUTURE 0
#   undef  JF_CAN_USE_OPENMP
#   define JF_CAN_USE_OPENMP 0
#elif defined(JF_USE_FUTURE)
#   undef  JF_CAN_USE_FUTURE
#   define JF_CAN_USE_FUTURE 1
#   undef  JF_CAN_USE_OPENMP
#   define JF_CAN_USE_OPENMP 0
#elif defined(JF_USE_OPENMP)
#   undef  JF_CAN_USE_FUTURE
#   define JF_CAN_USE_FUTURE 0
#   undef  JF_CAN_USE_OPENMP
#   define JF_CAN_USE_OPENMP 1
#else
#   define JF_CAN_USE_FUTURE 1
#   if __clang__
#   if __clang_major__ < 13
#   undef  JF_CAN_USE_FUTURE
#   define JF_CAN_USE_FUTURE 0
#   endif
#   endif

#   ifdef _OPENMP
#   define JF_CAN_USE_OPENMP 1
#   else
#   define JF_CAN_USE_OPENMP 0
#   endif
#endif

#if JF_CAN_USE_FUTURE
#include "threadpool.h"
namespace jf {
inline size_t get_parallel_threads() { return get_num_threads(); }
inline size_t set_parallel_threads(int nthreads) { return set_num_threads(nthreads); }
inline std::string get_parallel_backend() { return "native"; }
}
#elif JF_CAN_USE_OPENMP
// #pragma cling load("libomp").
#include <omp.h>
namespace jf {
inline size_t get_parallel_threads() { return omp_get_max_threads(); }
inline size_t set_parallel_threads(int nthreads)
{
    omp_set_num_threads(nthreads);
    return omp_get_max_threads();
}
inline std::string get_parallel_backend() { return "omp"; }
}
#else
namespace jf {
inline size_t get_parallel_threads() { return 1; }
inline size_t set_parallel_threads(int nthreads) { return 1; }
inline std::string get_parallel_backend() { return "none"; }
}
#endif


namespace jf {
namespace internal {

//#if 0
//    // used with _set_in_parallel_region to mark master thread
//    // as in parallel region while executing parallel primitives
//    thread_local bool in_parallel_region_ = false;
//
//    // thread number (task_id) set by parallel primitive
//    thread_local int thread_num_ = 0;
//
//    inline void _set_in_parallel_region(bool in_region) {
//      in_parallel_region_ = in_region;
//    }
//
//    inline void _unset_thread_num() {
//      thread_num_ = 0;
//    }
//#endif

    inline long divup(long x, long y) {
      return (x + y - 1) / y;
    }

    inline std::tuple<size_t, size_t> calc_num_tasks_and_chunk_size(
        long begin, long end, long grain_size)
    {
        if ((end - begin) < grain_size)
            return std::make_tuple(1, std::max((long)0, end - begin));
        // Choose number of tasks based on grain size and number of threads.
        size_t chunk_size = divup((end - begin), get_parallel_threads());
        // Make sure each task is at least grain_size size.
        chunk_size = std::max((size_t)grain_size, chunk_size);
        size_t num_tasks = divup((end - begin), chunk_size);
        return std::make_tuple(num_tasks, chunk_size);
    }

//#if 0
//    inline void set_thread_num(int thread_num) {
//      thread_num_ = thread_num;
//    }
//
//    inline int get_thread_num() {
//        return thread_num_;
//    }
//
//    inline bool in_parallel_region() {
//        return in_parallel_region_;
//    }
//
//    // RAII guard helps to support in_parallel_region() and get_thread_num() API.
//    struct ParallelRegionGuard {
//        ParallelRegionGuard(int task_id) {
//            internal::set_thread_num(task_id);
//            _set_in_parallel_region(true);
//        }
//
//        ~ParallelRegionGuard() {
//            _set_in_parallel_region(false);
//            _unset_thread_num();
//        }
//    };
//
//    struct ThreadIdGuard {
//      ThreadIdGuard(int new_id) : old_id_(get_thread_num()) {
//        set_thread_num(new_id);
//      }
//
//      ~ThreadIdGuard() {
//        set_thread_num(old_id_);
//      }
//
//     private:
//      int old_id_;
//    };
//#endif


#if JF_CAN_USE_FUTURE
    using future_type = std::future<void>;

    inline
    void invoke_parallel(long begin, long end, long grain_size,
                         const std::function<void(long, long)>& f) {
        size_t num_tasks, chunk_size;
        std::tie(num_tasks, chunk_size) =
            calc_num_tasks_and_chunk_size(begin, end, grain_size);

        auto pool = get_global_pool();
        std::queue<future_type> futures;

        // Build a task that processes chunk_size data points
        auto task = [f, begin, end, chunk_size](size_t task_id)
        {
            long local_start = begin + task_id * chunk_size;
            if (local_start < end) {
                long local_end = std::min(end, (long)(chunk_size + local_start));
                // ParallelRegionGuard guard(task_id);
                f(local_start, local_end);
            }
        };

        // Submit
        for (size_t task_id = 0; task_id < num_tasks; ++task_id)
            futures.push(pool->async(task, task_id));

        // Synchronize
        while (!futures.empty()) {
            futures.front().get();
            futures.pop();
        }
    }
#elif JF_CAN_USE_OPENMP
    inline
    void invoke_parallel(long begin, long end, long grain_size,
                         const std::function<void(long, long)>& f) {
        size_t num_tasks, chunk_size;
        std::tie(num_tasks, chunk_size) =
            calc_num_tasks_and_chunk_size(begin, end, grain_size);

        // Build a task that processes chunk_size data points
        auto task = [f, begin, end, chunk_size](size_t task_id)
        {
            long local_start = begin + task_id * chunk_size;
            if (local_start < end) {
                long local_end = std::min(end, (long)(chunk_size + local_start));
                // ParallelRegionGuard guard(task_id);
                f(local_start, local_end);
            }
        };

        // Parallel loop
#       pragma omp parallel for
        for (size_t task_id = 0; task_id < num_tasks; ++task_id)
            task(task_id);
    }
#else
    inline
    void invoke_parallel(long begin, long end, long grain_size,
                         const std::function<void(long, long)>& f)
    {
        return f(begin, end);
    }
#endif // JF_CAN_USE_FUTURE || JF_CAN_USE_OPENMP

} // namespace internal
} // namespace jf

#endif // JF_PARALLEL_IMPL_H
