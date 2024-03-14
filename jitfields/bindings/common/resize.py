
def get_shift_scale(anchor, inshape, outshape, factor):
    """Compute the shift and scale factor

    Parameters
    ----------
    anchor : {'edge', 'center', 'first'}
    inshape : list[int]
    outshape : list[int], optional
    factor : list[float], optional

    Returns
    -------
    shift : list[float]
    scale : list[float]

    """
    anchor = anchor[0].lower()
    if anchor == 'e':
        shift = 0.5
        scale = [si / so for si, so in zip(inshape, outshape)]
    elif anchor == 'c':
        shift = 0
        scale = [(si - 1) / (so - 1) for si, so in zip(inshape, outshape)]
    else:
        shift = 0
        scale = [1/f for f in factor]
    return shift, scale
