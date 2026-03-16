#pragma once

#include <vector_types.h>
#include <vector>
#include <string>
#include "shared_defs.h"

struct GeometryData {
  std::vector<float3> vertices;
  std::vector<float3> normals;
  std::vector<uint3> indices;
  std::vector<int> material_indices;
  std::vector<int> object_indices;
};

enum ObjectId {
  kObjectFloor = 0,
  kObjectCeiling = 1,
  kObjectBackWall = 2,
  kObjectLeftWall = 3,
  kObjectRightWall = 4,
  kObjectLight = 5,
  kObjectShortBox = 6,
  kObjectTallBox = 7,
  kObjectWater = 8,
  kObjectImportedGlass = 9,
  kObjectCount = 10,
};

// Math helpers (inline to avoid multiple definition issues if included elsewhere, 
// or we could move them to a common header. For now, scene.cpp will need them.)
float3 make_vec(float x, float y, float z);
float3 operator+(const float3& a, const float3& b);
float3 operator-(const float3& a, const float3& b);
float3 operator*(const float3& a, float s);
float3 operator/(const float3& a, float s);
float dot3(const float3& a, const float3& b);
float3 cross3(const float3& a, const float3& b);
float length3(const float3& v);
float3 normalize3(const float3& v);

struct SceneConfig {
  GeometryData geometry;
  std::vector<Material> materials;
  std::vector<Light> lights;
  float3 camera_pos;
  float3 camera_lookat;
  float3 camera_up;
  float camera_fov_y;
};

int add_vertex(GeometryData& geometry, float x, float y, float z, const float3& normal = {0.0f, 0.0f, 0.0f});
void add_triangle(GeometryData& geometry, int material, int object_id, int a, int b, int c);
void add_quad(GeometryData& geometry, int material, int object_id, int a, int b, int c, int d);
void add_box(GeometryData& geometry, const float3& min_corner, const float3& max_corner, int material, int object_id);
void add_ply_mesh(GeometryData& geometry, const std::string& filename, int material_id, int object_id, float3 scale, float3 translation);
void add_uv_sphere(GeometryData& geometry, const float3& center, float radius, int material, int object_id);

SceneConfig build_cornell_box_water(float light_intensity, float light_radius, unsigned int light_type, float reference_radius);
SceneConfig build_cornell_box_glass(float light_intensity, float light_radius, unsigned int light_type, float reference_radius);