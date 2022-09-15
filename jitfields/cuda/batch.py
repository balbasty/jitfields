import os

this_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_folder, 'batch.cuh'), 'rt') as f:
    code = f.read()
