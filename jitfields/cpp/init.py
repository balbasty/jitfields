import cppyy
import os

this_folder = os.path.abspath(os.path.dirname(__file__))
include_folder = os.path.join(this_folder, '..', 'csrc')
cppyy.add_include_path(include_folder)

sources = ['threadpool.cpp', 'parallel.cpp']
for source in sources:
    with open(os.path.join(include_folder, source)) as f:
        cppyy.cppdef(f.read())
