import cppyy
import os

this_folder = os.path.abspath(os.path.dirname(__file__))
cppyy.add_include_path(os.path.join(this_folder, '..', 'csrc'))
cppyy.include('parallel.h')

if hasattr(cppyy.gbl.jf, 'set_num_threads'):
    set_num_threads = cppyy.gbl.jf.set_num_threads
    get_num_threads = cppyy.gbl.jf.get_num_threads
else:
    def set_num_threads(x): return 1
    def get_num_threads(): return 1
