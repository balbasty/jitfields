#ifndef JF_TETRAHEDRON_LOOP
#define JF_TETRAHEDRON_LOOP


namespace jf {
namespace tetra {

template <typename scalar_t, typename index_t, typename offset_t>
void pull(scalar_t * out,
          const scalar_t * coord,
          const index_t * tetra,
          const scalar_t * values,
          int nbatch,
          const offset_t * size_tetra,    // (*batch, Ne, 4)
          const offset_t * size_coord,    // (*batch, Nv, 3)
          const offset_t * size_values,   // (*batch, Nv, Nc)
          const offset_t * size_out,      // (*batch, Nc, Nx, Ny, Nz)
          const offset_t * stride_tetra,
          const offset_t * stride_coord,
          const offset_t * stride_values,
          const offset_t * stride_out)
{
    offset_t nx  = size_out[nbatch];
    offset_t ny  = size_out[nbatch+1];
    offset_t nz  = size_out[nbatch+2];
    offset_t nc  = size_out[nbatch+3];
    offset_t osx = stride_out[nbatch];
    offset_t osy = stride_out[nbatch+1];
    offset_t osz = stride_out[nbatch+2];
    offset_t osc = stride_out[nbatch+3];
    offset_t vsn = stride_values[nbatch];
    offset_t vsc = stride_values[nbatch+1];
    offset_t fsn = stride_tetra[nbatch];
    offset_t fsk = stride_tetra[nbatch+1];
    offset_t csn = stride_coord[nbatch];
    offset_t csk = stride_coord[nbatch+1];

    offset_t numel = prod(size_tetra, nbatch+1);  // loop batch + tetra

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            const index_t * tetra_offset = tetra + index2offset(i, nbatch+1, size_tetra, stride_tetra);

            offset_t i0 = static_cast<offset_t>(*(tetra_offset));
            offset_t i1 = static_cast<offset_t>(*(tetra_offset + fsk));
            offset_t i2 = static_cast<offset_t>(*(tetra_offset + fsk*2));
            offset_t i3 = static_cast<offset_t>(*(tetra_offset + fsk*3));

            const scalar_t * coord_offset = coord + index2offset(i, nbatch, size_coord, stride_coord);
            scalar_t c0x = *(coord_offset + i0 * csn);
            scalar_t c0y = *(coord_offset + i0 * csn + csk);
            scalar_t c0z = *(coord_offset + i0 * csn + csk*2);
            scalar_t c1x = *(coord_offset + i1 * csn);
            scalar_t c1y = *(coord_offset + i1 * csn + csk);
            scalar_t c1z = *(coord_offset + i1 * csn + csk*2);
            scalar_t c2x = *(coord_offset + i2 * csn);
            scalar_t c2y = *(coord_offset + i2 * csn + csk);
            scalar_t c2z = *(coord_offset + i2 * csn + csk*2);
            scalar_t c3x = *(coord_offset + i3 * csn);
            scalar_t c3y = *(coord_offset + i3 * csn + csk);
            scalar_t c3z = *(coord_offset + i3 * csn + csk*2);

            const scalar_t * values_offset = values + index2offset(i, nbatch, size_values, stride_values);
            const scalar_t * v0 = values_offset + i0 * vsn;
            const scalar_t * v1 = values_offset + i1 * vsn;
            const scalar_t * v2 = values_offset + i2 * vsn;
            const scalar_t * v3 = values_offset + i3 * vsn;

            offset_t out_offset = index2offset(i, nbatch, size_out, stride_out);

            pull1(out + out_offset,
                  c0x, c0y, c0z, c1x, c1y, c1z, c2x, c2y, c2z, c3x, c3y, c3z,
                  v0, v1, v2, v3, nc, vsc, osc, nx, ny, nz, osx, osy, osz);
        }
    });
}

} // namespace tetra
} // namespace jf

#endif // JF_TETRAHEDRON_LOOP
