try:
    from .bindings.cpp.threads import set_num_threads, get_num_threads
except (ImportError, ModuleNotFoundError):
    pass
