#include "shared_defs.h"
#include "raytracer_helpers.h"

static __forceinline__ __device__ float3 sample_sphere(unsigned int& seed) {
    const float u = rnd(seed);
    const float v = rnd(seed);
    const float z = 2.0f * u - 1.0f;
    const float phi = 2.0f * CUDART_PI_F * v;
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return make_float3(r * cosf(phi), z, r * sinf(phi));
}

static __forceinline__ __device__ float3 sample_cone(unsigned int& seed, float cos_theta_max) {
    const float u = rnd(seed);
    const float v = rnd(seed);
    const float cos_theta = (1.0f - u) + u * cos_theta_max;
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    const float phi = 2.0f * CUDART_PI_F * v;
    return make_float3(sin_theta * cosf(phi), cos_theta, sin_theta * sinf(phi));
}

extern "C" __global__ void __raygen__emit_photons() {
  const uint3 idx = optixGetLaunchIndex();
  const unsigned int photon_index = idx.x;
  if (photon_index >= params.photon_launch_count) {
    return;
  }

  unsigned int seed =
      pcg_hash((photon_index + 1u) ^ (0x85ebca6bu * (params.iteration_index + 1u)) ^ 0xc2b2ae35u);

  if (params.light_count == 0) return;

  // Choose a light based on power
  // We compute it on the fly: pick randomly weighted by total emission power.
  
  float total_weight = 0.0f;
  for (unsigned int i = 0; i < params.light_count; ++i) {
      const Light l = params.lights[i];
      const bool is_point_light =
          (l.type == kLightTypePoint) ||
          (l.type == kLightTypeSphere && l.radius <= 1e-4f);
      if (is_point_light) {
              total_weight += luminance(l.emission) * 4.0f * CUDART_PI_F;
      } else if (l.type == kLightTypeSpot) {
              // Spot light power (simplified: same as point light but restricted to cone)
              // The energy is typically specified as intensity in the direction of the cone.
              // For simplicity, we'll treat it like a point light power-wise but only emit in cone.
              total_weight += luminance(l.emission) * 2.0f * CUDART_PI_F * (1.0f - l.v.x);
      } else if (l.type == kLightTypeSphere) {
              // Area light (emissive)
              float area = 4.0f * CUDART_PI_F * l.radius * l.radius;
              total_weight += luminance(l.emission) * area * CUDART_PI_F;
      } else {
          // Quad Light (emissive)
          float area = length3(cross3(l.u, l.v));
          total_weight += luminance(l.emission) * area * CUDART_PI_F;
      }
  }

  float r_light = rnd(seed) * total_weight;
  unsigned int light_idx = 0;
  float current_sum = 0.0f;
  float light_weight = 0.0f;
  for (unsigned int i = 0; i < params.light_count; ++i) {
      const Light l = params.lights[i];
      float weight = 0.0f;
      const bool is_point_light =
          (l.type == kLightTypePoint) ||
          (l.type == kLightTypeSphere && l.radius <= 1e-4f);
      if (is_point_light) {
              weight = luminance(l.emission) * 4.0f * CUDART_PI_F;
      } else if (l.type == kLightTypeSpot) {
              weight = luminance(l.emission) * 2.0f * CUDART_PI_F * (1.0f - l.v.x);
      } else if (l.type == kLightTypeSphere) {
              weight = luminance(l.emission) * (4.0f * CUDART_PI_F * l.radius * l.radius) * CUDART_PI_F;
      } else {
          weight = luminance(l.emission) * length3(cross3(l.u, l.v)) * CUDART_PI_F;
      }
      current_sum += weight;
      if (r_light <= current_sum || i == params.light_count - 1) {
          light_idx = i;
          light_weight = weight;
          break;
      }
  }

  const Light light = params.lights[light_idx];
  float3 origin;
  float3 direction;
  float light_pdf = 1.0f;

    if (light.type == kLightTypePoint || (light.type == kLightTypeSphere && light.radius <= 1e-4f)) {
        // Point Light
        origin = light.position;
        direction = sample_sphere(seed);
        light_pdf = 1.0f / (4.0f * CUDART_PI_F);
    } else if (light.type == kLightTypeSpot) {
        // Spot Light
        origin = light.position;
        const float3 local_dir = sample_cone(seed, light.v.x);
        float3 tangent, bitangent;
        const float3 spot_dir = normalize3(light.u);
        make_basis(spot_dir, tangent, bitangent);
        direction = normalize3(add3(add3(mul3s(tangent, local_dir.x), mul3s(spot_dir, local_dir.y)), mul3s(bitangent, local_dir.z)));
        light_pdf = 1.0f / (2.0f * CUDART_PI_F * (1.0f - light.v.x));
    } else if (light.type == kLightTypeSphere) {
        const float u = rnd(seed);
        const float v = rnd(seed);
        const float z = 2.0f * u - 1.0f;
        const float phi = 2.0f * CUDART_PI_F * v;
        const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float3 sphere_local_pos = make_float3(r * cosf(phi), z, r * sinf(phi));
        origin = add3(light.position, mul3s(sphere_local_pos, light.radius));
        direction = cosine_sample_hemisphere(seed, sphere_local_pos);
        light_pdf = 1.0f / CUDART_PI_F; // Pdf relative to projected area
  } else {
    // Quad Light
    const float u = rnd(seed);
    const float v = rnd(seed);
    origin = add3(light.position, add3(mul3s(light.u, u), mul3s(light.v, v)));
    const float3 normal = normalize3(cross3(light.u, light.v));
    direction = cosine_sample_hemisphere(seed, normal);
    light_pdf = 1.0f / CUDART_PI_F;
  }

  const float inv_pdf_selection = total_weight / fmaxf(light_weight, 1e-6f);
  const float inv_pdf_launch = 1.0f / static_cast<float>(params.photon_launch_count);
  
  // flux = Intensity * invPdfSelection * invPdfLaunch
  // For Area Light: Intensity = L * Area * PI
  // For Point Light: Intensity = I * 4PI
  float3 flux_weight = mul3s(light.emission, inv_pdf_selection * inv_pdf_launch);
  if (light.type == kLightTypePoint || (light.type == kLightTypeSphere && light.radius <= 1e-4f)) {
      flux_weight = mul3s(flux_weight, 4.0f * CUDART_PI_F);
  } else if (light.type == kLightTypeSpot) {
      flux_weight = mul3s(flux_weight, 2.0f * CUDART_PI_F * (1.0f - light.v.x));
  } else {
      float area = (light.type == kLightTypeSphere) ? (4.0f * CUDART_PI_F * light.radius * light.radius) : length3(cross3(light.u, light.v));
      flux_weight = mul3s(flux_weight, area * CUDART_PI_F);
  }

  PhotonTracePRD prd = {};
  prd.origin = add3(origin, mul3s(direction, 2.0f * kRayEpsilon));
  prd.direction = direction;
  prd.flux = flux_weight;
  prd.seed = seed;
  prd.depth = 0;
  prd.specular_chain_active = 0u;
  prd.done = 0;

  while (!prd.done && prd.depth < params.photon_max_bounces) {
    trace_scene_photon(&prd);
  }
}
