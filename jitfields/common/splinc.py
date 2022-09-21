import math as pymath


def get_poles(order):
    empty = []
    if order in (0, 1):
        return empty
    if order == 2:
        return [pymath.sqrt(8.) - 3.]
    if order == 3:
        return [pymath.sqrt(3.) - 2.]
    if order == 4:
        return [pymath.sqrt(664. - pymath.sqrt(438976.)) + pymath.sqrt(304.) - 19.,
                pymath.sqrt(664. + pymath.sqrt(438976.)) - pymath.sqrt(304.) - 19.]
    if order == 5:
        return [pymath.sqrt(67.5 - pymath.sqrt(4436.25)) + pymath.sqrt(26.25) - 6.5,
                pymath.sqrt(67.5 + pymath.sqrt(4436.25)) - pymath.sqrt(26.25) - 6.5]
    if order == 6:
        return [-0.488294589303044755130118038883789062112279161239377608394,
                -0.081679271076237512597937765737059080653379610398148178525368,
                -0.00141415180832581775108724397655859252786416905534669851652709]
    if order == 7:
        return [-0.5352804307964381655424037816816460718339231523426924148812,
                -0.122554615192326690515272264359357343605486549427295558490763,
                -0.0091486948096082769285930216516478534156925639545994482648003]
    raise NotImplementedError
