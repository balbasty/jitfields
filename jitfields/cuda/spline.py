import os

this_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_folder, 'spline.cuh'), 'rt') as f:
    code = f.read()

cnames = ['Nearest', 'Linear', 'Quadratic', 'Cubic',
          'FourthOrder', 'FifthOrder', 'SixthOrder', 'SeventhOrder']


convert_order = {
    'nearest': 0,
    'linear': 1,
    'quadratic': 2,
    'cubic': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
}