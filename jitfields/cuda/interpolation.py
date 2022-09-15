import os

this_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_folder, 'interpolation.cuh'), 'rt') as f:
    code = f.read()

cnames = ['Nearest', 'Linear', 'Quadratic', 'Cubic',
          'FourthOrder', 'FifthOrder', 'SixthOrder', 'SeventhOrder']
