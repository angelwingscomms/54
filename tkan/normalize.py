def normalize(xmin, xmax, *arrays):
    scale = xmax - xmin + 1e-8
    return tuple((arr - xmin) / scale if getattr(arr, 'ndim', 0) == 3 else arr for arr in arrays)
