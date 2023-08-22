#ifndef JF_THREADPOOL_INL
#define JF_THREADPOOL_INL
#include <memory>

namespace jf {
namespace internal {
    // Some of this is copied from pytorch/aten

    size_t get_env_num_threads(const char* var_name, size_t def_value = 0) {
        try {
            if (auto* value = std::getenv(var_name)) {
                size_t nthreads = static_cast<size_t>(std::stoi(value));
                if (nthreads) return nthreads;
            }
        } catch (const std::exception& e) {}
        return def_value;
    }

    size_t default_num_threads_from_hardware() {
        auto num_threads = std::thread::hardware_concurrency();
#       if defined(_M_X64) || defined(__x86_64__)
            num_threads /= 2;
#       endif
        return num_threads;
    }

    size_t default_num_threads() {
        size_t nthreads = get_env_num_threads("JF_NUM_THREADS", 0);
        if (nthreads == 0) nthreads = get_env_num_threads("OMP_NUM_THREADS", nthreads);
        if (nthreads == 0) nthreads = get_env_num_threads("MKL_NUM_THREADS", nthreads);
        if (nthreads == 0) nthreads = default_num_threads_from_hardware();
        if (nthreads == 0) nthreads = 1;
        return nthreads;
    }

    int num_threads = default_num_threads();
    std::shared_ptr<ThreadPool> global_pool(nullptr);
} // namespace internal

size_t set_num_threads(size_t nthreads) {
    if (nthreads == 0) nthreads = 1; 
    size_t old_num_threads = internal::num_threads;
    internal::num_threads = nthreads;
    if (old_num_threads != internal::num_threads)
        internal::global_pool.reset(new ThreadPool(internal::num_threads));
    return internal::num_threads;
}

inline size_t get_num_threads() {
    return internal::num_threads;
}

std::shared_ptr<ThreadPool> get_global_pool() {
    if (!internal::global_pool)
        internal::global_pool.reset(new ThreadPool(internal::num_threads));
    return internal::global_pool;
}

} // namespace jf
#endif // JF_THREADPOOL_INL
