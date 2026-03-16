#include <optix_device.h>

extern "C" __global__ void __anyhit__scene_shadow() {
  const SceneHitGroupData* hit_group_data =
      reinterpret_cast<const SceneHitGroupData*>(optixGetSbtDataPointer());
  const unsigned int primitive_index = optixGetPrimitiveIndex();
  const Material material =
      hit_group_data->materials[hit_group_data->material_indices[primitive_index]];
  if (material.type == kMaterialDielectric) {
    optixIgnoreIntersection();
  }
}

extern "C" __global__ void __closesthit__scene_radiance() {
  const SceneHitGroupData* hit_group_data =
      reinterpret_cast<const SceneHitGroupData*>(optixGetSbtDataPointer());
  RadiancePRD* prd = get_prd<RadiancePRD>();

  const unsigned int primitive_index = optixGetPrimitiveIndex();
  const uint3 tri = hit_group_data->indices[primitive_index];
  const float2 bary = optixGetTriangleBarycentrics();
  const float b0 = 1.0f - bary.x - bary.y;
  const float b1 = bary.x;
  const float b2 = bary.y;

  const float3 p0 = hit_group_data->vertices[tri.x];
  const float3 p1 = hit_group_data->vertices[tri.y];
  const float3 p2 = hit_group_data->vertices[tri.z];
  const float3 hit_point = add3(add3(mul3s(p0, b0), mul3s(p1, b1)), mul3s(p2, b2));
  const float3 geometric_normal = normalize3(cross3(sub3(p1, p0), sub3(p2, p0)));
  float3 surface_normal = geometric_normal;
  if (hit_group_data->normals) {
    const float3 n0 = hit_group_data->normals[tri.x];
    const float3 n1 = hit_group_data->normals[tri.y];
    const float3 n2 = hit_group_data->normals[tri.z];
    surface_normal = normalize3(add3(add3(mul3s(n0, b0), mul3s(n1, b1)), mul3s(n2, b2)));
  }

  const Material material =
      hit_group_data->materials[hit_group_data->material_indices[primitive_index]];
  const float3 ray_dir = optixGetWorldRayDirection();

  float3 shading_normal = surface_normal;
  if (dot3(shading_normal, ray_dir) > 0.0f) {
    shading_normal = neg3(shading_normal);
  }

  float3 face_normal = geometric_normal;
  if (dot3(face_normal, ray_dir) > 0.0f) {
    face_normal = neg3(face_normal);
  }

  if (luminance(material.emission) > 0.0f) {
    prd->result = mul3(prd->throughput, material.emission);
    prd->done = 1;
    return;
  }

  if (material.type == kMaterialDielectric) {
    const bool front_face = dot3(ray_dir, surface_normal) < 0.0f;
    const float eta_i = front_face ? 1.0f : material.ior;
    const float eta_t = front_face ? material.ior : 1.0f;
    const float eta = eta_i / eta_t;
    const float cos_theta = fminf(dot3(neg3(ray_dir), shading_normal), 1.0f);
    const float reflectance = fresnel_schlick(cos_theta, eta_i, eta_t);

    float3 next_direction;
    if (rnd(prd->seed) < reflectance || !refract3(ray_dir, shading_normal, eta, next_direction)) {
      next_direction = reflect3(ray_dir, shading_normal);
    }

    prd->throughput = mul3(prd->throughput, material.color);
    prd->origin = add3(hit_point, mul3s(next_direction, kRayEpsilon));
    prd->direction = next_direction;
    ++prd->depth;

    if (luminance(prd->throughput) < 1.0e-4f) {
      prd->result = make_float3(0.0f, 0.0f, 0.0f);
      prd->done = 1;
    }
    return;
  }

  float3 caustic_indirect = make_float3(0.0f, 0.0f, 0.0f);
  float3 global_indirect = make_float3(0.0f, 0.0f, 0.0f);
  unsigned int caustic_count = 0;
  unsigned int global_count = 0;

  gather_photons(hit_point,
                 shading_normal,
                 face_normal,
                 material.color,
                 ray_dir,
                 params.caustic_photon_handle,
                 params.caustic_photon_gather_scale,
                 kPhotonGatherCausticRayType,
                 prd->seed,
                 caustic_indirect,
                 caustic_count);
  gather_photons(hit_point,
                 shading_normal,
                 face_normal,
                 material.color,
                 ray_dir,
                 params.global_photon_handle,
                 params.global_photon_gather_scale,
                 kPhotonGatherGlobalRayType,
                 prd->seed,
                 global_indirect,
                 global_count);

  const float3 indirect = add3(caustic_indirect, global_indirect);
  const unsigned int gathered_count = caustic_count + global_count;
  const float3 direct = compute_direct_lighting(hit_point, shading_normal, material.color, prd->seed);

  float3 surface_radiance = add3(direct, indirect);
  if (params.render_mode == kRenderModePhotonGatherCount) {
    const float v = fminf(static_cast<float>(gathered_count) / 12.0f, 1.0f);
    surface_radiance = make_float3(v, 0.25f * v, 1.0f - v);
  } else if (params.render_mode == kRenderModeIndirectOnly) {
    surface_radiance = indirect;
  } else if (params.render_mode == kRenderModeDirectOnly) {
    surface_radiance = direct;
  } else if (params.render_mode == kRenderModeCausticOnly) {
    surface_radiance = caustic_indirect;
  }

  prd->result = mul3(prd->throughput, surface_radiance);
  prd->done = 1;
}

extern "C" __global__ void __closesthit__scene_photon() {
  const SceneHitGroupData* hit_group_data =
      reinterpret_cast<const SceneHitGroupData*>(optixGetSbtDataPointer());
  PhotonTracePRD* prd = get_prd<PhotonTracePRD>();

  const unsigned int primitive_index = optixGetPrimitiveIndex();
  const uint3 tri = hit_group_data->indices[primitive_index];
  const float2 bary = optixGetTriangleBarycentrics();
  const float b0 = 1.0f - bary.x - bary.y;
  const float b1 = bary.x;
  const float b2 = bary.y;

  const float3 p0 = hit_group_data->vertices[tri.x];
  const float3 p1 = hit_group_data->vertices[tri.y];
  const float3 p2 = hit_group_data->vertices[tri.z];
  const float3 hit_point = add3(add3(mul3s(p0, b0), mul3s(p1, b1)), mul3s(p2, b2));
  const float3 geometric_normal = normalize3(cross3(sub3(p1, p0), sub3(p2, p0)));
  float3 surface_normal = geometric_normal;
  if (hit_group_data->normals) {
    const float3 n0 = hit_group_data->normals[tri.x];
    const float3 n1 = hit_group_data->normals[tri.y];
    const float3 n2 = hit_group_data->normals[tri.z];
    surface_normal = normalize3(add3(add3(mul3s(n0, b0), mul3s(n1, b1)), mul3s(n2, b2)));
  }

  float3 shading_normal = surface_normal;
  if (dot3(shading_normal, prd->direction) > 0.0f) {
    shading_normal = neg3(shading_normal);
  }

  float3 face_normal = geometric_normal;
  if (dot3(face_normal, prd->direction) > 0.0f) {
    face_normal = neg3(face_normal);
  }

  const unsigned int object_id =
      static_cast<unsigned int>(hit_group_data->object_indices[primitive_index]);
  const Material material =
      hit_group_data->materials[hit_group_data->material_indices[primitive_index]];


  if (material.type == kMaterialDielectric) {
    const bool front_face = dot3(prd->direction, surface_normal) < 0.0f;
    const float eta_i = front_face ? 1.0f : material.ior;
    const float eta_t = front_face ? material.ior : 1.0f;
    const float eta = eta_i / eta_t;
    const float cos_theta = fminf(dot3(neg3(prd->direction), shading_normal), 1.0f);
    const float reflectance = fresnel_schlick(cos_theta, eta_i, eta_t);

    float3 next_direction;
    if (rnd(prd->seed) < reflectance || !refract3(prd->direction, shading_normal, eta, next_direction)) {
      next_direction = reflect3(prd->direction, shading_normal);
    }

    update_specular_chain_after_specular(prd);
    prd->origin = add3(hit_point, mul3s(next_direction, kRayEpsilon));
    prd->direction = next_direction;
    ++prd->depth;
    return;
  }

  const bool store_caustic = prd->specular_chain_active != 0u;
  const bool accept_global =
      !store_caustic &&
      (params.global_photon_rejection >= 1.0f || rnd(prd->seed) <= params.global_photon_rejection);
  if (store_caustic || accept_global) {
    Photon* photon_buffer = store_caustic ? params.caustic_photons : params.global_photons;
    unsigned int* photon_counter =
        store_caustic ? params.caustic_photon_count : params.global_photon_count;

    const unsigned int slot = atomicAdd(photon_counter, 1u);
    if (slot < params.max_photons) {
      Photon photon;
      photon.position = hit_point;
      photon.face_normal = face_normal;
      photon.incident_dir = prd->direction;
      photon.flux = prd->flux;
      if (!store_caustic && params.global_photon_rejection > 0.0f) {
        photon.flux = div3s(photon.flux, params.global_photon_rejection);
      }
      photon.object_id = object_id;
      photon.specular_path = store_caustic ? 1u : 0u;
      photon_buffer[slot] = photon;
    }
  }

  prd->flux = mul3(prd->flux, material.color);
  if (luminance(prd->flux) < 1.0e-4f) {
    prd->done = 1;
    return;
  }

  const float rr_val = fminf(luminance(prd->flux), 0.95f);
  if (rnd(prd->seed) > rr_val) {
    prd->done = 1;
    return;
  }
  prd->flux = div3s(prd->flux, rr_val);

  update_specular_chain_after_diffuse(prd);
  prd->origin = add3(hit_point, mul3s(shading_normal, kRayEpsilon));
  prd->direction = cosine_sample_hemisphere(prd->seed, shading_normal);
  ++prd->depth;
}
