#pragma once

#include "shared_defs.h"

void build_photon_aabbs(const Photon* d_photons,
                        OptixAabb* d_aabbs,
                        unsigned int count,
                        float radius);

void clear_float4_buffer(float4* d_buffer, unsigned int count);

void accumulate_float4_buffer(const float4* d_src, float4* d_dst, unsigned int count);

void scale_float4_buffer(float4* d_buffer, unsigned int count, float scale);
