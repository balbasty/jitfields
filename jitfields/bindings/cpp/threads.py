import cppyy
from .utils import include

include()
cppyy.include('../lib/parallel.h')

if hasattr(getattr(cppyy.gbl, 'jf', None), 'set_parallel_threads'):
    set_num_threads = cppyy.gbl.jf.set_parallel_threads
    get_num_threads = cppyy.gbl.jf.get_parallel_threads
    def get_parallel_backend():
        return str(cppyy.gbl.jf.get_parallel_backend())
else:
    def set_num_threads(x): return 1
    def get_num_threads(): return 1
    def get_parallel_backend(): return "none"

