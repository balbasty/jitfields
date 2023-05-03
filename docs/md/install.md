# Installation

Installation through pip should work, although I don't know how robust the cupy/pytorch 
interaction is in term of cuda version.
```sh
pip install git+https://github.com/balbasty/jitfields
```

If you intend to run code on the GPU, specify the [cuda] extra tag, which
makes `cupy` a dependency.
```sh
pip install "jitfields[cuda] @ git+https://github.com/balbasty/jitfields"
```

Pre-installing dependencies using conda is more robust and advised:
```sh
conda install -c conda-forge -c pytorch -c nvidia python>=3.6 numpy cupy ccpyy pytorch>=1.8 cudatoolkit=11.1
pip install "jitfields[cuda] @ git+https://github.com/balbasty/jitfields"
```

