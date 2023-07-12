# ================================================
#       Utilities to load FS surfaces
# ================================================

import nibabel.freesurfer.io as fsio
import torch
import numpy as np


_np_to_torch_dtype = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex32,
    np.complex128: torch.complex64,
    np.complex256: torch.complex128,
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    # upcast
    np.uint16: torch.int32,
    np.uint32: torch.int64,
    np.uint64: torch.int64, # risk overflow
}


def default_affine(shape, voxel_size=1, **backend):
    """Generate a RAS affine matrix

    Parameters
    ----------
    shape : list[int]
        Lattice shape
    voxel_size : [sequence of] float
        Voxel size
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    affine : (D+1, D+1) tensor
        Affine matrix

    """
    ndim = len(shape)
    aff = torch.eye(ndim+1, **backend)
    backend = dict(dtype=aff.dtype, device=aff.device)

    # set voxel size
    voxel_size = torch.as_tensor(voxel_size, **backend).flatten()
    pad = max(0, ndim - len(voxel_size))
    pad = [voxel_size[-1:]] * pad
    voxel_size = torch.cat([voxel_size, *pad])
    voxel_size = voxel_size[:ndim]
    aff[:-1, :-1] *= voxel_size[None, :]

    # set center fov
    shape = torch.as_tensor(shape, **backend)
    aff[:-1, -1] = -voxel_size * (shape - 1) / 2

    return aff


def load_mesh(fname, return_space=False, numpy=False):
    """Load a mesh in memory

    Parameters
    ----------
    fname : str
        Path to surface file

    return_space : bool, default=False
        Return the affine matrix and shape of the original volume

    numpy : bool, default=False
        Return numpy array instead of torch tensor

    Returns
    -------
    coord : (N, D) tensor
        Node coordinates.
        Each node has a coordinate in an ambient space.

    faces : (M, K) tensor
        Faces.
        Each face is made of K nodes, whose indices are stored in this tensor.
        For triangular meshes, K = 3.

    affine : (D+1, D+1) tensor, if `return_space`
        Mapping from the `coord`'s ambient space to a standard space.
        In Freesurfer surfaces, edges coordinates are also expressed in
        voxels of the original volumetric file, in which case the affine
        maps these voxel coordinates to millimetric RAS coordinates.

    shape : (D,) list[int], if `return_space`
        Shape of the original volume.

    """
    c, f, *meta = fsio.read_geometry(fname, read_metadata=return_space)

    if not numpy:
        if not np.dtype(c.dtype).isnative:
            c = c.newbyteorder().byteswap(inplace=True)
        if not np.dtype(f.dtype).isnative:
            f = f.newbyteorder().byteswap(inplace=True)
        c = torch.as_tensor(c, dtype=_np_to_torch_dtype[np.dtype(c.dtype).type])
        f = torch.as_tensor(f, dtype=_np_to_torch_dtype[np.dtype(f.dtype).type])

    if not return_space:
        return c, f

    shape = None
    if 'volume' in meta:
        shape = torch.as_tensor(meta['volume']).tolist()
    aff = torch.eye(c.shape[-1])
    if 'cras' in meta:
        x, y, z, c = meta['xras'], meta['yras'], meta['zras'], meta['cras']
        x, y, z, c = torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(z), torch.as_tensor(c)
        aff[:-1, :] = torch.stack([x, y, z, c], dim=1).to(aff)
    if 'voxelsize' in meta:
        vx = torch.as_tensor(meta['voxelsize'], dtype=torch.float32)
        aff[:-1, :-1] *= vx[None, :]

    return c, f, aff, shape


# ================================================
#       The actual code
# ================================================


from jitfields.distance import mesh_distance_signed
import torch
import matplotlib.pyplot as plt

fname = '/autofs/cluster/vxmdata1/FS_Slim/orig/OASIS/OAS1_0001_MR1/surf/lh.white'
vertices, faces, aff, shape = load_mesh(fname, return_space=True)

coord = torch.stack(torch.meshgrid(
    [torch.arange((mn-5).floor(), (mx+5).floor()) 
     for mn, mx in zip(vertices.min(0).values, vertices.max(0).values)]), -1)
coord = coord.to(vertices)

coord = coord.cuda()
sdist = mesh_distance_signed(coord, vertices, faces)


mid = sdist.shape[1]//2
plt.subplot(3, 1, 1)
plt.imshow(sdist[:, mid, :].cpu() < 0, interpolation='nearest')
plt.colorbar()
plt.subplot(3, 1, 2)
plt.imshow(sdist[:, mid, :].cpu(), cmap='coolwarm', vmin=-45, vmax=45, interpolation='nearest')
plt.colorbar()
plt.subplot(3, 1, 3)
plt.imshow(sdist[:, mid, :].abs().cpu(), cmap='viridis', vmin=0, vmax=45, interpolation='nearest')
plt.colorbar()
plt.show()


