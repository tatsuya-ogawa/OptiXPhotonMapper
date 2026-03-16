#pragma once

#include <optix_device.h>
#include <math_constants.h>
#include <vector_functions.h>

extern "C" __constant__ Params params;

namespace {

constexpr unsigned int kStochasticPhotonListSize = 3;
constexpr float kPhotonCollectTMin = 1.0e-6f;
constexpr float kPhotonCollectTMax = 2.0e-6f;
constexpr float kRayEpsilon = 1.0e-3f;

struct RadiancePRD {
  float3 result;
  float3 throughput;
  float3 origin;
  float3 direction;
  unsigned int seed;
  unsigned int depth;
  int done;
};

struct ShadowPRD {
  unsigned int visible;
};

struct PhotonTracePRD {
  float3 origin;
  float3 direction;
  float3 flux;
  unsigned int seed;
  unsigned int depth;
  unsigned int specular_chain_active;
  int done;
};

struct GatherPRD {
  float3 surface_normal;
  float3 face_normal;
  float3 surface_color;
  float3 accumulated_radiance;
  unsigned int candidate_count;
  unsigned int unique_count;
  unsigned int seed;
  int photon_ids[kStochasticPhotonListSize];
  float3 sampled_radiance[kStochasticPhotonListSize];
};

static __forceinline__ __device__ float3 add3(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 sub3(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 neg3(const float3& a) {
  return make_float3(-a.x, -a.y, -a.z);
}

static __forceinline__ __device__ float3 mul3(const float3& a, const float3& b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __device__ float3 mul3s(const float3& a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float3 div3s(const float3& a, float s) {
  return make_float3(a.x / s, a.y / s, a.z / s);
}

static __forceinline__ __device__ float dot3(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 cross3(const float3& a, const float3& b) {
  return make_float3(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float3 reflect3(const float3& v, const float3& n) {
  return sub3(v, mul3s(n, 2.0f * dot3(v, n)));
}

static __forceinline__ __device__ float length3(const float3& v) {
  return sqrtf(dot3(v, v));
}

static __forceinline__ __device__ float3 normalize3(const float3& v) {
  return div3s(v, fmaxf(length3(v), 1.0e-8f));
}

static __forceinline__ __device__ unsigned int pcg_hash(unsigned int input) {
  unsigned int state = input * 747796405u + 2891336453u;
  unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

static __forceinline__ __device__ unsigned int pcg(unsigned int& state) {
  state = state * 747796405u + 2891336453u;
  unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

static __forceinline__ __device__ float rnd(unsigned int& state) {
  return static_cast<float>(pcg(state) & 0x00ffffff) / static_cast<float>(0x01000000);
}

static __forceinline__ __device__ void pack_pointer(void* ptr, unsigned int& p0, unsigned int& p1) {
  const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
  p0 = static_cast<unsigned int>(uptr >> 32);
  p1 = static_cast<unsigned int>(uptr & 0xffffffffu);
}

static __forceinline__ __device__ void* unpack_pointer(unsigned int p0, unsigned int p1) {
  const unsigned long long uptr =
      (static_cast<unsigned long long>(p0) << 32) | static_cast<unsigned long long>(p1);
  return reinterpret_cast<void*>(uptr);
}

template <typename T>
static __forceinline__ __device__ T* get_prd() {
  return reinterpret_cast<T*>(unpack_pointer(optixGetPayload_0(), optixGetPayload_1()));
}

static __forceinline__ __device__ void make_basis(const float3& n, float3& tangent, float3& bitangent) {
  if (fabsf(n.x) > 0.1f) {
    tangent = normalize3(cross3(make_float3(0.0f, 1.0f, 0.0f), n));
  } else {
    tangent = normalize3(cross3(make_float3(1.0f, 0.0f, 0.0f), n));
  }
  bitangent = cross3(n, tangent);
}

static __forceinline__ __device__ float3 cosine_sample_hemisphere(unsigned int& seed,
                                                                  const float3& normal) {
  const float u1 = rnd(seed);
  const float u2 = rnd(seed);
  const float r = sqrtf(u1);
  const float phi = 2.0f * CUDART_PI_F * u2;
  const float x = r * cosf(phi);
  const float z = r * sinf(phi);
  const float y = sqrtf(fmaxf(0.0f, 1.0f - u1));

  float3 tangent;
  float3 bitangent;
  make_basis(normal, tangent, bitangent);
  return normalize3(add3(add3(mul3s(tangent, x), mul3s(normal, y)), mul3s(bitangent, z)));
}

static __forceinline__ __device__ float luminance(const float3& c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static __forceinline__ __device__ bool refract3(const float3& v,
                                                const float3& n,
                                                float eta,
                                                float3& refracted) {
  const float cos_theta = fminf(dot3(neg3(v), n), 1.0f);
  const float sin2_theta = fmaxf(0.0f, 1.0f - cos_theta * cos_theta);
  const float k = 1.0f - eta * eta * sin2_theta;
  if (k < 0.0f) {
    return false;
  }
  const float3 r_out_perp = mul3s(add3(v, mul3s(n, cos_theta)), eta);
  const float3 r_out_parallel = mul3s(n, -sqrtf(k));
  refracted = normalize3(add3(r_out_perp, r_out_parallel));
  return true;
}

static __forceinline__ __device__ float fresnel_schlick(float cos_theta, float eta_i, float eta_t) {
  const float r0_term = (eta_i - eta_t) / (eta_i + eta_t);
  const float r0 = r0_term * r0_term;
  const float m = 1.0f - cos_theta;
  return r0 + (1.0f - r0) * m * m * m * m * m;
}

static __forceinline__ __device__ void trace_scene_radiance(RadiancePRD* prd) {
  unsigned int p0;
  unsigned int p1;
  pack_pointer(prd, p0, p1);
  optixTrace(params.scene_handle,
             prd->origin,
             prd->direction,
             2.0e-2f,
             1.0e16f,
             0.0f,
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             kRadianceRayType,
             kRayTypeCount,
             kRadianceRayType,
             p0,
             p1);
}

static __forceinline__ __device__ bool trace_scene_shadow(const float3& origin,
                                                          const float3& direction,
                                                          float max_distance) {
  ShadowPRD prd;
  prd.visible = 0;
  unsigned int p0;
  unsigned int p1;
  pack_pointer(&prd, p0, p1);
  optixTrace(params.scene_handle,
             origin,
             direction,
             2.0e-2f,
             max_distance,
             0.0f,
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
             kShadowRayType,
             kRayTypeCount,
             kShadowRayType,
             p0,
             p1);
  return prd.visible != 0;
}

static __forceinline__ __device__ void trace_scene_photon(PhotonTracePRD* prd) {
  unsigned int p0;
  unsigned int p1;
  pack_pointer(prd, p0, p1);
  optixTrace(params.scene_handle,
             prd->origin,
             prd->direction,
             2.0e-2f,
             1.0e16f,
             0.0f,
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             kPhotonTraceRayType,
             kRayTypeCount,
             kPhotonTraceRayType,
             p0,
             p1);
}

static __forceinline__ __device__ void trace_photon_gather(const float3& origin,
                                                           const float3& direction,
                                                           OptixTraversableHandle handle,
                                                           unsigned int ray_type,
                                                           GatherPRD* prd) {
  if (!handle) {
    return;
  }

  unsigned int p0;
  unsigned int p1;
  pack_pointer(prd, p0, p1);
  optixTrace(handle,
             origin,
             direction,
             kPhotonCollectTMin,
             kPhotonCollectTMax,
             0.0f,
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_NONE,
             ray_type,
             kRayTypeCount,
             ray_type,
             p0,
             p1);
}

static __forceinline__ __device__ float3 compute_direct_lighting(const float3& hit_point,
                                                                 const float3& shading_normal,
                                                                 const float3& surface_color,
                                                                 unsigned int& seed) {
  float3 total_direct = make_float3(0.0f, 0.0f, 0.0f);
  const unsigned int samples_per_light = params.direct_light_samples > 0 ? params.direct_light_samples : 1u;

  for (unsigned int l_idx = 0; l_idx < params.light_count; ++l_idx) {
    const Light light = params.lights[l_idx];
    float3 direct = make_float3(0.0f, 0.0f, 0.0f);

    const bool is_point_light =
        (light.type == kLightTypePoint) ||
        (light.type == kLightTypeSphere && light.radius <= 1.0e-4f);

    if (is_point_light) {
      const float3 to_light = sub3(light.position, hit_point);
      const float distance_sq = dot3(to_light, to_light);
      const float distance = sqrtf(distance_sq);
      if (distance > 1.0e-4f) {
        const float3 light_dir = div3s(to_light, distance);
        const float n_dot_l = fmaxf(dot3(shading_normal, light_dir), 0.0f);
        if (n_dot_l > 0.0f) {
          if (trace_scene_shadow(add3(hit_point, mul3s(shading_normal, kRayEpsilon)),
                                 light_dir,
                                 distance - 2.0e-3f)) {
            direct = add3(direct,
                          mul3(surface_color,
                               mul3s(light.emission,
                                     n_dot_l /
                                         (CUDART_PI_F * fmaxf(distance_sq, 1.0e-6f)))));
          }
        }
      }
    } else if (light.type == kLightTypeSphere) {
      const float light_area = 4.0f * CUDART_PI_F * light.radius * light.radius;
      for (unsigned int sample_idx = 0; sample_idx < samples_per_light; ++sample_idx) {
        const float u = rnd(seed);
        const float v = rnd(seed);
        const float z = 2.0f * u - 1.0f;
        const float phi = 2.0f * CUDART_PI_F * v;
        const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float3 sphere_normal = make_float3(r * cosf(phi), z, r * sinf(phi));
        const float3 light_point = add3(light.position, mul3s(sphere_normal, light.radius));

        const float3 to_light = sub3(light_point, hit_point);
        const float distance_sq = dot3(to_light, to_light);
        const float distance = sqrtf(distance_sq);
        if (distance <= 1.0e-4f) continue;

        const float3 light_dir = div3s(to_light, distance);
        const float n_dot_l = fmaxf(dot3(shading_normal, light_dir), 0.0f);
        if (n_dot_l <= 0.0f) continue;

        const float light_cos = fmaxf(dot3(neg3(light_dir), sphere_normal), 0.0f);
        if (light_cos <= 0.0f) continue;

        if (trace_scene_shadow(add3(hit_point, mul3s(shading_normal, kRayEpsilon)), light_dir, distance - 2.0e-3f)) {
          const float geometry = (n_dot_l * light_cos) / fmaxf(distance_sq, 1.0e-6f);
          direct = add3(direct, mul3(surface_color, mul3s(light.emission, light_area * geometry / CUDART_PI_F)));
        }
      }
    } else if (light.type == kLightTypeSpot) {
      const float3 to_light = sub3(light.position, hit_point);
      const float distance_sq = dot3(to_light, to_light);
      const float distance = sqrtf(distance_sq);
      if (distance > 1.0e-4f) {
        const float3 light_dir = div3s(to_light, distance);
        const float n_dot_l = fmaxf(dot3(shading_normal, light_dir), 0.0f);
        if (n_dot_l > 0.0f) {
          const float3 spot_dir = normalize3(light.u);
          const float cos_theta = dot3(neg3(light_dir), spot_dir);
          if (cos_theta > light.v.x) {
            if (trace_scene_shadow(add3(hit_point, mul3s(shading_normal, kRayEpsilon)),
                                   light_dir,
                                   distance - 2.0e-3f)) {
              float falloff = 1.0f;
              if (light.v.y > 0.0f) {
                  falloff = powf(cos_theta, light.v.y);
              }
              // Normalization: point light intensity scaled by falloff
              direct = add3(direct,
                            mul3(surface_color,
                                 mul3s(light.emission,
                                       (falloff * n_dot_l) /
                                           (CUDART_PI_F * fmaxf(distance_sq, 1.0e-6f)))));
            }
          }
        }
      }
    } else {
      // Quad Light
      const float3 normal = normalize3(cross3(light.u, light.v));
      const float light_area = length3(cross3(light.u, light.v));
      for (unsigned int sample_idx = 0; sample_idx < samples_per_light; ++sample_idx) {
        const float u = rnd(seed);
        const float v = rnd(seed);
        const float3 light_point = add3(light.position, add3(mul3s(light.u, u), mul3s(light.v, v)));

        const float3 to_light = sub3(light_point, hit_point);
        const float distance_sq = dot3(to_light, to_light);
        const float distance = sqrtf(distance_sq);
        if (distance <= 1.0e-4f) continue;

        const float3 light_dir = div3s(to_light, distance);
        const float n_dot_l = fmaxf(dot3(shading_normal, light_dir), 0.0f);
        if (n_dot_l <= 0.0f) continue;

        const float light_cos = fmaxf(dot3(neg3(light_dir), normal), 0.0f);
        if (light_cos <= 0.0f) continue;

        if (trace_scene_shadow(add3(hit_point, mul3s(shading_normal, kRayEpsilon)), light_dir, distance - 2.0e-3f)) {
          const float geometry = (n_dot_l * light_cos) / fmaxf(distance_sq, 1.0e-6f);
          direct = add3(direct, mul3(surface_color, mul3s(light.emission, light_area * geometry / CUDART_PI_F)));
        }
      }
    }
    total_direct = add3(total_direct, div3s(direct, static_cast<float>(samples_per_light)));
  }

  return total_direct;
}

static __forceinline__ __device__ void gather_photons(const float3& hit_point,
                                                      const float3& shading_normal,
                                                      const float3& face_normal,
                                                      const float3& surface_color,
                                                      const float3& ray_dir,
                                                      OptixTraversableHandle handle,
                                                      float gather_scale,
                                                      unsigned int ray_type,
                                                      unsigned int& seed,
                                                      float3& indirect,
                                                      unsigned int& gathered_count) {
  GatherPRD gather = {};
  gather.surface_normal = shading_normal;
  gather.face_normal = face_normal;
  gather.surface_color = surface_color;
  gather.accumulated_radiance = make_float3(0.0f, 0.0f, 0.0f);
  gather.candidate_count = 0;
  gather.unique_count = 0;
  gather.seed = seed;
  for (unsigned int i = 0; i < kStochasticPhotonListSize; ++i) {
    gather.photon_ids[i] = -1;
    gather.sampled_radiance[i] = make_float3(0.0f, 0.0f, 0.0f);
  }

  if (!handle) {
    indirect = make_float3(0.0f, 0.0f, 0.0f);
    gathered_count = 0;
    return;
  }

  trace_photon_gather(hit_point, ray_dir, handle, ray_type, &gather);

  float3 radiance = gather.accumulated_radiance;
  if (params.photon_eval_mode == kPhotonEvalStochastic && gather.candidate_count > gather.unique_count &&
      gather.unique_count > 0) {
    radiance = mul3s(radiance,
                     static_cast<float>(gather.candidate_count) /
                         static_cast<float>(gather.unique_count));
  }

  // Apply area weight: 1 / (PI * R^2)
  indirect = mul3s(radiance, gather_scale);
  gathered_count = params.photon_eval_mode == kPhotonEvalFull ? gather.candidate_count : gather.unique_count;
  seed = gather.seed;
}

static __forceinline__ __device__ void update_specular_chain_after_diffuse(PhotonTracePRD* prd) {
  prd->specular_chain_active = 0u;
}

static __forceinline__ __device__ void update_specular_chain_after_specular(PhotonTracePRD* prd) {
  if (params.use_multi_diffuse_caustic_map != 0u) {
    prd->specular_chain_active = 1u;
  } else if (prd->depth == 0u) {
    prd->specular_chain_active = 1u;
  }
}

} // namespace
