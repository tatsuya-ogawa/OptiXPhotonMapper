#include "shared_defs.h"
#include "raytracer_helpers.h"

extern "C" __global__ void __raygen__render() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  const unsigned int pixel_index = idx.y * dim.x + idx.x;
  unsigned int seed = pcg_hash(pixel_index ^ (0x9e3779b9u * (params.iteration_index + 1u)) ^ 0xa511e9b3u);
  float3 accum = make_float3(0.0f, 0.0f, 0.0f);

  for (unsigned int sample = 0; sample < params.samples_per_pixel; ++sample) {
    const float jitter_x = rnd(seed);
    const float jitter_y = rnd(seed);
    const float d_x = (static_cast<float>(idx.x) + jitter_x) / static_cast<float>(dim.x);
    const float d_y = (static_cast<float>(idx.y) + jitter_y) / static_cast<float>(dim.y);
    const float screen_x = 2.0f * d_x - 1.0f;
    const float screen_y = 1.0f - 2.0f * d_y;
    const float3 ray_dir =
        normalize3(add3(params.camera_w,
                        add3(mul3s(params.camera_u, screen_x), mul3s(params.camera_v, screen_y))));

    RadiancePRD prd = {};
    prd.result = make_float3(0.0f, 0.0f, 0.0f);
    prd.throughput = make_float3(1.0f, 1.0f, 1.0f);
    prd.origin = params.camera_position;
    prd.direction = ray_dir;
    prd.seed = seed;
    prd.depth = 0;
    prd.done = 0;

    while (!prd.done && prd.depth < params.photon_max_bounces) {
      trace_scene_radiance(&prd);
    }

    seed = prd.seed;
    accum = add3(accum, prd.result);
  }

  const float3 color = div3s(accum, static_cast<float>(params.samples_per_pixel));
  params.image[idx.y * params.width + idx.x] = make_float4(color.x, color.y, color.z, 1.0f);
}

extern "C" __global__ void __miss__radiance() {
  RadiancePRD* prd = get_prd<RadiancePRD>();
  prd->result = make_float3(0.0f, 0.0f, 0.0f);
  prd->done = 1;
}

extern "C" __global__ void __miss__shadow() {
  ShadowPRD* prd = get_prd<ShadowPRD>();
  prd->visible = 1;
}

extern "C" __global__ void __miss__photon_trace() {
  PhotonTracePRD* prd = get_prd<PhotonTracePRD>();
  prd->done = 1;
}

extern "C" __global__ void __miss__photon_gather() {
}

extern "C" __global__ void __intersection__photon() {
  const PhotonHitGroupData* hit_group_data =
      reinterpret_cast<const PhotonHitGroupData*>(optixGetSbtDataPointer());
  const Photon photon = hit_group_data->photons[optixGetPrimitiveIndex()];
  const float3 delta = sub3(optixGetObjectRayOrigin(), photon.position);
  const float radius = hit_group_data->radius;
  if (dot3(delta, delta) <= radius * radius) {
    optixReportIntersection(optixGetRayTmin(), 0);
  }
}

extern "C" __global__ void __anyhit__photon_gather() {
  const PhotonHitGroupData* hit_group_data =
      reinterpret_cast<const PhotonHitGroupData*>(optixGetSbtDataPointer());
  GatherPRD* prd = get_prd<GatherPRD>();
  const int photon_id = static_cast<int>(optixGetPrimitiveIndex());

  if (params.photon_eval_mode == kPhotonEvalStochastic) {
    for (unsigned int i = 0; i < prd->unique_count; ++i) {
      if (prd->photon_ids[i] == photon_id) {
        optixIgnoreIntersection();
        return;
      }
    }
  }

  const Photon photon = hit_group_data->photons[photon_id];
  if (params.photon_normal_reject_cos > 0.0f &&
      dot3(prd->face_normal, photon.face_normal) < params.photon_normal_reject_cos) {
    optixIgnoreIntersection();
    return;
  }

  const float n_dot_l = dot3(prd->surface_normal, neg3(photon.incident_dir));
  if (n_dot_l <= 0.0f) {
    optixIgnoreIntersection();
    return;
  }

  // Radiance Estimate: (BSDF eval / NdotL) * Flux
  // For diffuse material: BSDF eval = color * NdotL / PI
  // f_r = (color * NdotL / PI) / NdotL = color / PI
  const float3 f_r = mul3s(prd->surface_color, 1.0f / CUDART_PI_F);
  const float3 contribution = mul3(f_r, photon.flux);
  ++prd->candidate_count;

  if (params.photon_eval_mode == kPhotonEvalFull) {
    prd->accumulated_radiance = add3(prd->accumulated_radiance, contribution);
    ++prd->unique_count;
  } else {
    if (prd->unique_count < kStochasticPhotonListSize) {
      prd->photon_ids[prd->unique_count] = photon_id;
      prd->sampled_radiance[prd->unique_count] = contribution;
      prd->accumulated_radiance = add3(prd->accumulated_radiance, contribution);
      ++prd->unique_count;
    } else {
      const unsigned int replace_index =
          static_cast<unsigned int>(rnd(prd->seed) * static_cast<float>(prd->candidate_count));
      if (replace_index < kStochasticPhotonListSize) {
        prd->accumulated_radiance =
            sub3(prd->accumulated_radiance, prd->sampled_radiance[replace_index]);
        prd->sampled_radiance[replace_index] = contribution;
        prd->photon_ids[replace_index] = photon_id;
        prd->accumulated_radiance = add3(prd->accumulated_radiance, contribution);
      }
    }
  }

  optixIgnoreIntersection();
}

#include "raytracer.cu"
#include "scene_ray_gen.cu"

