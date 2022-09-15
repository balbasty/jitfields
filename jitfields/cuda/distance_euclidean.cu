using namespace std;

template <typename out_t, typename in_t>
__device__ __forceinline__
out_t mycast(in_t x) { return static_cast<out_t>(static_cast<float>(x)); }

template <typename offset_t, typename scalar_t>
__device__
scalar_t intersection(scalar_t * f, offset_t * v, scalar_t w2,
                      offset_t k, offset_t q,
                      offset_t size, offset_t stride_buf)
{
    offset_t vk = v[k * stride_buf];
    scalar_t fvk = f[vk * stride_buf];
    scalar_t fq = f[q * stride_buf];
    offset_t a = q - vk, b = q + vk;
    scalar_t s = fq - mycast<scalar_t>(fvk);
    s += w2 * mycast<scalar_t>(a * b);
    s /= mycast<scalar_t>(2) * w2 * mycast<scalar_t>(a);
    return s;
}

template <typename offset_t, typename scalar_t>
__device__
void fillin(scalar_t * f, offset_t * v, scalar_t * z, scalar_t * d, scalar_t w2,
            offset_t size, offset_t stride, offset_t stride_buf)
{
    offset_t k = 0;
    offset_t vk;
    z += stride_buf;
    for (offset_t q = 0; q < size; ++q) {
        scalar_t fq = mycast<scalar_t>(q);
        while ((k < size-1) && (*z < fq)) {
            z += stride_buf;
            ++k;
        }
        vk = v[k * stride_buf];
        f[q * stride] = d[vk * stride_buf]
                      + w2 * mycast<scalar_t>(square(q - vk));
    }
}

template <typename offset_t, typename scalar_t>
__device__
void algo(scalar_t * f, offset_t * v, scalar_t * z, scalar_t * d, scalar_t w2,
          offset_t size, offset_t stride, offset_t stride_buf)
{
    if (size == 1) return;

    for (offset_t q = 0; q < size; ++q, f += stride, d += stride_buf)
        *d = *f;
    f -= size * stride;
    d -= size * stride_buf;

    v[0] = 0;
    z[0] = -(1./0.);
    z[stride_buf] = 1./0.;
    scalar_t s;
    offset_t k = 0;
    scalar_t * zk;
    for (offset_t q=1; q < size; ++q) {
        zk = z + k * stride_buf;
        while (1) {
            s = intersection(d, v, w2, k, q, size, stride_buf);
            if ((k == 0) || (s > *zk))
                break;
            --k;
            zk -= stride_buf;
        }

        ++k;
        v[k * stride_buf] = q;
        z[k * stride_buf] = s;
        if (k < size-1)
            z[(k+1) * stride_buf] = 1./0.;
    }
    fillin(f, v, z, d, w2, size, stride, stride_buf);
}


template <typename scalar_t, typename offset_t>
__global__ void kernel(scalar_t * f, char * buf, scalar_t w, int ndim,
                       const offset_t * size, const offset_t *  stride)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size, ndim-1);

    offset_t n = size[ndim-1];
    offset_t stride_buf = blockDim.x * gridDim.x;
    offset_t * v = reinterpret_cast<offset_t *>(buf);
    scalar_t * z = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * sizeof(offset_t));
    scalar_t * d = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * (sizeof(offset_t) + sizeof(scalar_t)));

    w = w*w;

    for (offset_t i=index; index < nthreads; index += stride_buf, i=index)
    {
        offset_t batch_offset = index2offset(i, ndim-1, size, stride);
        algo(f + batch_offset, v + index, z + index, d + index, w,
             n, stride[ndim-1], stride_buf);
    }
}