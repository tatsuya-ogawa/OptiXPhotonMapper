#include "photon_helpers.h"

#include <cuda_runtime.h>

namespace {

__global__ void build_photon_aabbs_kernel(const Photon* photons,
                                          OptixAabb* aabbs,
                                          unsigned int count,
                                          float radius) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const Photon photon = photons[idx];
  OptixAabb aabb;
  aabb.minX = photon.position.x - radius;
  aabb.minY = photon.position.y - radius;
  aabb.minZ = photon.position.z - radius;
  aabb.maxX = photon.position.x + radius;
  aabb.maxY = photon.position.y + radius;
  aabb.maxZ = photon.position.z + radius;
  aabbs[idx] = aabb;
}

__global__ void clear_float4_buffer_kernel(float4* buffer, unsigned int count) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  buffer[idx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}

__global__ void accumulate_float4_buffer_kernel(const float4* src, float4* dst, unsigned int count) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  dst[idx].x += src[idx].x;
  dst[idx].y += src[idx].y;
  dst[idx].z += src[idx].z;
  dst[idx].w = 1.0f;
}

__global__ void scale_float4_buffer_kernel(float4* buffer, unsigned int count, float scale) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  buffer[idx].x *= scale;
  buffer[idx].y *= scale;
  buffer[idx].z *= scale;
}

}  // namespace

void build_photon_aabbs(const Photon* d_photons,
                        OptixAabb* d_aabbs,
                        unsigned int count,
                        float radius) {
  if (count == 0) {
    return;
  }

  const unsigned int block_size = 256;
  const unsigned int grid_size = (count + block_size - 1) / block_size;
  build_photon_aabbs_kernel<<<grid_size, block_size>>>(d_photons, d_aabbs, count, radius);
}

void clear_float4_buffer(float4* d_buffer, unsigned int count) {
  if (count == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int grid_size = (count + block_size - 1) / block_size;
  clear_float4_buffer_kernel<<<grid_size, block_size>>>(d_buffer, count);
}

void accumulate_float4_buffer(const float4* d_src, float4* d_dst, unsigned int count) {
  if (count == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int grid_size = (count + block_size - 1) / block_size;
  accumulate_float4_buffer_kernel<<<grid_size, block_size>>>(d_src, d_dst, count);
}

void scale_float4_buffer(float4* d_buffer, unsigned int count, float scale) {
  if (count == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int grid_size = (count + block_size - 1) / block_size;
  scale_float4_buffer_kernel<<<grid_size, block_size>>>(d_buffer, count, scale);
}
