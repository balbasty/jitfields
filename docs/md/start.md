# Getting started

We use a functional API and try to follow most of PyTorch's convention.
Check out the API and start playing.

Notable differences include the way sampling coordinates are encoded 
(`[0, N-1]` instead of `[-1, 1]`, no index flipping), and the fact that 
we often use a "channel last" dimension ordering.

