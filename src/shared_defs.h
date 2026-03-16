#pragma once

#include <optix.h>
#include <vector_types.h>

constexpr unsigned int kRadianceRayType = 0;
constexpr unsigned int kShadowRayType = 1;
constexpr unsigned int kPhotonTraceRayType = 2;
constexpr unsigned int kPhotonGatherCausticRayType = 3;
constexpr unsigned int kPhotonGatherGlobalRayType = 4;
constexpr unsigned int kRayTypeCount = 5;

constexpr unsigned int kMaxGatheredPhotons = 64;

constexpr unsigned int kRenderModeBeauty = 0;
constexpr unsigned int kRenderModePhotonGatherCount = 1;
constexpr unsigned int kRenderModeIndirectOnly = 2;
constexpr unsigned int kRenderModeDirectOnly = 3;
constexpr unsigned int kRenderModeCausticOnly = 4;

constexpr unsigned int kPhotonEvalStochastic = 0;
constexpr unsigned int kPhotonEvalFull = 1;

constexpr unsigned int kMaterialDiffuse = 0;
constexpr unsigned int kMaterialDielectric = 1;

constexpr unsigned int kLightTypeSphere = 0;
constexpr unsigned int kLightTypeQuad = 1;
constexpr unsigned int kLightTypePoint = 2;
constexpr unsigned int kLightTypeSpot = 3;

constexpr unsigned int kObjectGlassSphereId = 8;

struct Material {
  float3 color;
  float3 emission;
  unsigned int type;
  float ior;
};

struct Photon {
  float3 position;
  float3 face_normal;
  float3 incident_dir;
  float3 flux;
  unsigned int object_id;
  unsigned int specular_path;
};

struct Light {
  unsigned int type;
  float3 position;
  float3 u;  // For Area: base vector 1. For Spot: direction.
  float3 v;  // For Area: base vector 2. For Spot: v.x = opening angle (cos), v.y = exponent.
  float3 emission;
  float radius;
};

struct Params {
  float4* image;
  unsigned int width;
  unsigned int height;
  unsigned int samples_per_pixel;
  unsigned int iteration_index;

  unsigned int photon_launch_count;
  unsigned int photon_max_bounces;
  unsigned int max_photons;
  unsigned int direct_light_samples;
  unsigned int photon_eval_mode;
  unsigned int render_mode;
  unsigned int use_multi_diffuse_caustic_map;

  float global_photon_gather_scale;
  float caustic_photon_gather_scale;
  float photon_normal_reject_cos;
  float global_photon_rejection;

  Photon* global_photons;
  Photon* caustic_photons;
  unsigned int* global_photon_count;
  unsigned int* caustic_photon_count;

  OptixTraversableHandle scene_handle;
  OptixTraversableHandle global_photon_handle;
  OptixTraversableHandle caustic_photon_handle;

  Light* lights;
  unsigned int light_count;

  float3 camera_position;
  float3 camera_u;
  float3 camera_v;
  float3 camera_w;
};

struct SceneHitGroupData {
  const float3* vertices;
  const float3* normals;
  const uint3* indices;
  const int* material_indices;
  const int* object_indices;
  const Material* materials;
};

struct PhotonHitGroupData {
  const Photon* photons;
  float radius;
};
