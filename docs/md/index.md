# jitfields

This package implements functions for (large) tensors that cannot be efficiently 
implemented in pure PyTorch (or with some performance penalty). These functions 
are instead implemented in pure C++ and/or CUDA, and compiled just-in-time 
using [`cppyy`](https://github.com/wlav/cppyy) (for CPU/C++) or 
[`cupy`](https://github.com/cupy/cupy) (for GPU/CUDA).

As long as both these packages are properly installed, `jitfields` is a pure 
python package and does not requires any compilation at setup (contrary to 
most pytorch extensions). 

We currently implement:

- Euclidean distance transforms (substitute for `scipy.ndimage.distance_transform_edt`).
- B-spline sampling/interpolation up to order 7 (substitute for `scipy.ndimage.zoom`, 
  `scipy.ndimage.spline_filter`, `scipy.ndimage.map_coordinates`).
- Finite difference regularisers for dense displacement fields and vector fields.
- Linear algebra operations for batches of symmetric matrices with compact storage.

In the near future, we plan to implement convolution routines that support many 
boundary conditions without requiring the volume to be padded and reallocated under 
the hood (contrary to what PyTorch currently does).

## Target audience

This package does not aim to implement all useful functions for ND tensors, 
but only those that cannot be implemented efficiently in pure PyTorch. It is 
therefore not intended to be used by a wide audience, but only by advanced users.

I am currently implementing higher level APIs, that exposes jitfields functions 
but also a range of pure PyTorch utilities and high-level tools, within the 
[`nitorch`](https://github.com/nitorch) package suite. Reach out if you're 
interested.

