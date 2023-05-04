try:
    from .bindings.cpp.threads import set_num_threads, get_num_threads, get_parallel_backend
except (ImportError, ModuleNotFoundError):
    from warnings import warn

    def set_num_threads(*args, **kwargs):
        warn('Could not import C++ bindings', ImportWarning)

    def get_num_threads(*args, **kwargs):
        warn('Could not import C++ bindings', ImportWarning)

    def get_parallel_backend(*args, **kwargs):
        warn('Could not import C++ bindings', ImportWarning)
