#include "scene.h"
#include "shared_defs.h"
#include <optix.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>

float3 make_vec(float x, float y, float z) {
  return make_float3(x, y, z);
}

float3 operator+(const float3& a, const float3& b) {
  return make_vec(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 operator-(const float3& a, const float3& b) {
  return make_vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 operator*(const float3& a, float s) {
  return make_vec(a.x * s, a.y * s, a.z * s);
}

float3 operator/(const float3& a, float s) {
  return make_vec(a.x / s, a.y / s, a.z / s);
}

float dot3(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 cross3(const float3& a, const float3& b) {
  return make_vec(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

float length3(const float3& v) {
  return std::sqrt(dot3(v, v));
}

float3 normalize3(const float3& v) {
  return v / length3(v);
}

int add_vertex(GeometryData& geometry, float x, float y, float z, const float3& normal) {
  geometry.vertices.push_back(make_vec(x, y, z));
  geometry.normals.push_back(normal);
  return static_cast<int>(geometry.vertices.size() - 1);
}

void add_triangle(GeometryData& geometry, int material, int object_id, int a, int b, int c) {
  geometry.indices.push_back(make_uint3(a, b, c));
  geometry.material_indices.push_back(material);
  geometry.object_indices.push_back(object_id);
}

void add_quad(GeometryData& geometry, int material, int object_id, int a, int b, int c, int d) {
  add_triangle(geometry, material, object_id, a, b, c);
  add_triangle(geometry, material, object_id, a, c, d);
}

void add_box(GeometryData& geometry,
                    const float3& min_corner,
                    const float3& max_corner,
                    int material,
                    int object_id) {
  // Front face (fixed z = min_corner.z)
  {
    const float3 n = make_vec(0.0f, 0.0f, -1.0f);
    const int v0 = add_vertex(geometry, min_corner.x, min_corner.y, min_corner.z, n);
    const int v1 = add_vertex(geometry, max_corner.x, min_corner.y, min_corner.z, n);
    const int v2 = add_vertex(geometry, max_corner.x, max_corner.y, min_corner.z, n);
    const int v3 = add_vertex(geometry, min_corner.x, max_corner.y, min_corner.z, n);
    add_quad(geometry, material, object_id, v0, v1, v2, v3);
  }
  // Back face (fixed z = max_corner.z)
  {
    const float3 n = make_vec(0.0f, 0.0f, 1.0f);
    const int v0 = add_vertex(geometry, min_corner.x, min_corner.y, max_corner.z, n);
    const int v1 = add_vertex(geometry, min_corner.x, max_corner.y, max_corner.z, n);
    const int v2 = add_vertex(geometry, max_corner.x, max_corner.y, max_corner.z, n);
    const int v3 = add_vertex(geometry, max_corner.x, min_corner.y, max_corner.z, n);
    add_quad(geometry, material, object_id, v0, v1, v2, v3);
  }
  // Bottom face (fixed y = min_corner.y)
  {
    const float3 n = make_vec(0.0f, -1.0f, 0.0f);
    const int v0 = add_vertex(geometry, min_corner.x, min_corner.y, min_corner.z, n);
    const int v1 = add_vertex(geometry, min_corner.x, min_corner.y, max_corner.z, n);
    const int v2 = add_vertex(geometry, max_corner.x, min_corner.y, max_corner.z, n);
    const int v3 = add_vertex(geometry, max_corner.x, min_corner.y, min_corner.z, n);
    add_quad(geometry, material, object_id, v0, v1, v2, v3);
  }
  // Top face (fixed y = max_corner.y)
  {
    const float3 n = make_vec(0.0f, 1.0f, 0.0f);
    const int v0 = add_vertex(geometry, min_corner.x, max_corner.y, min_corner.z, n);
    const int v1 = add_vertex(geometry, max_corner.x, max_corner.y, min_corner.z, n);
    const int v2 = add_vertex(geometry, max_corner.x, max_corner.y, max_corner.z, n);
    const int v3 = add_vertex(geometry, min_corner.x, max_corner.y, max_corner.z, n);
    add_quad(geometry, material, object_id, v0, v1, v2, v3);
  }
  // Left face (fixed x = min_corner.x)
  {
    const float3 n = make_vec(-1.0f, 0.0f, 0.0f);
    const int v0 = add_vertex(geometry, min_corner.x, min_corner.y, min_corner.z, n);
    const int v1 = add_vertex(geometry, min_corner.x, max_corner.y, min_corner.z, n);
    const int v2 = add_vertex(geometry, min_corner.x, max_corner.y, max_corner.z, n);
    const int v3 = add_vertex(geometry, min_corner.x, min_corner.y, max_corner.z, n);
    add_quad(geometry, material, object_id, v0, v1, v2, v3);
  }
  // Right face (fixed x = max_corner.x)
  {
    const float3 n = make_vec(1.0f, 0.0f, 0.0f);
    const int v0 = add_vertex(geometry, max_corner.x, min_corner.y, min_corner.z, n);
    const int v1 = add_vertex(geometry, max_corner.x, min_corner.y, max_corner.z, n);
    const int v2 = add_vertex(geometry, max_corner.x, max_corner.y, max_corner.z, n);
    const int v3 = add_vertex(geometry, max_corner.x, max_corner.y, min_corner.z, n);
    add_quad(geometry, material, object_id, v0, v1, v2, v3);
  }
}

void add_ply_mesh(GeometryData& geometry,
                         const std::string& filename,
                         int material_id,
                         int object_id,
                         float3 scale,
                         float3 translation) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open PLY file: " << filename << "\n";
        return;
    }
    std::string header;
    int vertex_count = 0;
    int face_count = 0;
    int property_count = 0;
    bool has_normals = false;
    
    char c;
    while (f.get(c)) {
        header += c;
        if (header.size() >= 10 && header.substr(header.size() - 10) == "end_header") {
            while (f.get(c) && c != '\n');
            break;
        }
    }

    size_t v_pos = header.find("element vertex");
    if (v_pos != std::string::npos) {
        vertex_count = std::stoi(header.substr(v_pos + 15));
    }
    size_t f_pos = header.find("element face");
    if (f_pos != std::string::npos) {
        face_count = std::stoi(header.substr(f_pos + 13));
    }
    
    // Count properties and check for normals
    size_t prop_elem_pos = header.find("element vertex");
    size_t next_elem_pos = header.find("element", prop_elem_pos + 1);
    std::string vertex_header = header.substr(prop_elem_pos, next_elem_pos - prop_elem_pos);
    
    size_t pos = 0;
    while ((pos = vertex_header.find("property float", pos)) != std::string::npos) {
        property_count++;
        std::string prop_line = vertex_header.substr(pos, 30);
        if (prop_line.find("nx") != std::string::npos) has_normals = true;
        pos += 14;
    }

    int start_v = static_cast<int>(geometry.vertices.size());
    std::vector<float3> local_vertices;
    
    float3 min_p = make_vec(1e30f, 1e30f, 1e30f);
    float3 max_p = make_vec(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < vertex_count; ++i) {
        std::vector<float> v_vals(property_count);
        f.read(reinterpret_cast<char*>(v_vals.data()), sizeof(float) * property_count);
        if (!f) break;
        float3 pos = make_vec(v_vals[0] * scale.x + translation.x,
                              v_vals[1] * scale.y + translation.y,
                              v_vals[2] * scale.z + translation.z);
        geometry.vertices.push_back(pos);
        local_vertices.push_back(pos);
        min_p.x = fminf(min_p.x, pos.x); min_p.y = fminf(min_p.y, pos.y); min_p.z = fminf(min_p.z, pos.z);
        max_p.x = fmaxf(max_p.x, pos.x); max_p.y = fmaxf(max_p.y, pos.y); max_p.z = fmaxf(max_p.z, pos.z);
        if (has_normals && property_count >= 6) {
            geometry.normals.push_back(normalize3(make_vec(v_vals[3], v_vals[4], v_vals[5])));
        } else {
            geometry.normals.push_back(make_vec(0.0f, 0.0f, 0.0f)); 
        }
    }
    std::cout << "Mesh World Bounds: Min=(" << min_p.x << "," << min_p.y << "," << min_p.z << ") Max=(" << max_p.x << "," << max_p.y << "," << max_p.z << ")\n";

    for (int i = 0; i < face_count; ++i) {
        uint8_t count;
        f.read(reinterpret_cast<char*>(&count), 1);
        if (!f) break;
        std::vector<int> indices(count);
        f.read(reinterpret_cast<char*>(indices.data()), count * sizeof(int));
        if (count == 3) {
            geometry.indices.push_back(make_uint3(start_v + indices[0],
                                                  start_v + indices[1],
                                                  start_v + indices[2]));
            geometry.material_indices.push_back(material_id);
            geometry.object_indices.push_back(object_id);
            
            if (!has_normals) {
                float3 v0 = local_vertices[indices[0]];
                float3 v1 = local_vertices[indices[1]];
                float3 v2 = local_vertices[indices[2]];
                float3 n = normalize3(cross3(v1 - v0, v2 - v0));
                geometry.normals[start_v + indices[0]] = geometry.normals[start_v + indices[0]] + n;
                geometry.normals[start_v + indices[1]] = geometry.normals[start_v + indices[1]] + n;
                geometry.normals[start_v + indices[2]] = geometry.normals[start_v + indices[2]] + n;
            }
        }
    }
    
    if (!has_normals) {
        for (int i = 0; i < vertex_count; ++i) {
            float3& n = geometry.normals[start_v + i];
            if (length3(n) > 1e-6f) n = normalize3(n);
            else n = make_vec(0.0f, 1.0f, 0.0f);
        }
    }
    
    std::cout << "Loaded PLY mesh: " << filename << " (" << vertex_count << " vertices, " << face_count << " faces, props=" << property_count << ")\n";
}

void add_uv_sphere(GeometryData& geometry,
                          const float3& center,
                          float radius,
                          int material,
                          int object_id) {
  constexpr int kSlices = 40;
  constexpr int kStacks = 20;
  constexpr float kPi = 3.1415926535f;
  std::vector<std::vector<int>> ring_vertices(kStacks + 1, std::vector<int>(kSlices + 1));

  for (int stack = 0; stack <= kStacks; ++stack) {
    const float v = static_cast<float>(stack) / static_cast<float>(kStacks);
    const float phi = kPi * v;
    const float y = cosf(phi);
    const float r = sinf(phi);
    for (int slice = 0; slice <= kSlices; ++slice) {
      const float u = static_cast<float>(slice) / static_cast<float>(kSlices);
      const float theta = 2.0f * kPi * u;
      const float x = r * cosf(theta);
      const float z = r * sinf(theta);
      ring_vertices[stack][slice] =
          add_vertex(geometry, center.x + radius * x, center.y + radius * y, center.z + radius * z, make_vec(x, y, z));
    }
  }

  for (int stack = 0; stack < kStacks; ++stack) {
    for (int slice = 0; slice < kSlices; ++slice) {
      const int v00 = ring_vertices[stack][slice];
      const int v01 = ring_vertices[stack][slice + 1];
      const int v10 = ring_vertices[stack + 1][slice];
      const int v11 = ring_vertices[stack + 1][slice + 1];

      if (stack == 0) {
        add_triangle(geometry, material, object_id, v00, v10, v11);
      } else if (stack == kStacks - 1) {
        add_triangle(geometry, material, object_id, v00, v10, v01);
      } else {
        add_triangle(geometry, material, object_id, v00, v10, v11);
        add_triangle(geometry, material, object_id, v00, v11, v01);
      }
    }
  }
}

namespace {
void add_cornell_box_walls(GeometryData& geometry) {
  constexpr int kWhite = 0;
  constexpr int kRed = 1;
  constexpr int kGreen = 2;

  const float3 floor_n = make_vec(0.0f, 1.0f, 0.0f);
  const int floor0 = add_vertex(geometry, 0.0f, 0.0f, 0.0f, floor_n);
  const int floor1 = add_vertex(geometry, 555.0f, 0.0f, 0.0f, floor_n);
  const int floor2 = add_vertex(geometry, 555.0f, 0.0f, 555.0f, floor_n);
  const int floor3 = add_vertex(geometry, 0.0f, 0.0f, 555.0f, floor_n);
  add_quad(geometry, kWhite, kObjectFloor, floor0, floor1, floor2, floor3);

  const float3 ceil_n = make_vec(0.0f, -1.0f, 0.0f);
  const int ceil0 = add_vertex(geometry, 0.0f, 555.0f, 0.0f, ceil_n);
  const int ceil1 = add_vertex(geometry, 0.0f, 555.0f, 555.0f, ceil_n);
  const int ceil2 = add_vertex(geometry, 555.0f, 555.0f, 555.0f, ceil_n);
  const int ceil3 = add_vertex(geometry, 555.0f, 555.0f, 0.0f, ceil_n);
  add_quad(geometry, kWhite, kObjectCeiling, ceil0, ceil1, ceil2, ceil3);

  const float3 back_n = make_vec(0.0f, 0.0f, -1.0f);
  const int back0 = add_vertex(geometry, 0.0f, 0.0f, 555.0f, back_n);
  const int back1 = add_vertex(geometry, 555.0f, 0.0f, 555.0f, back_n);
  const int back2 = add_vertex(geometry, 555.0f, 555.0f, 555.0f, back_n);
  const int back3 = add_vertex(geometry, 0.0f, 555.0f, 555.0f, back_n);
  add_quad(geometry, kWhite, kObjectBackWall, back0, back1, back2, back3);

  const float3 left_n = make_vec(1.0f, 0.0f, 0.0f);
  const int left0 = add_vertex(geometry, 0.0f, 0.0f, 0.0f, left_n);
  const int left1 = add_vertex(geometry, 0.0f, 0.0f, 555.0f, left_n);
  const int left2 = add_vertex(geometry, 0.0f, 555.0f, 555.0f, left_n);
  const int left3 = add_vertex(geometry, 0.0f, 555.0f, 0.0f, left_n);
  add_quad(geometry, kRed, kObjectLeftWall, left0, left1, left2, left3);

  const float3 right_n = make_vec(-1.0f, 0.0f, 0.0f);
  const int right0 = add_vertex(geometry, 555.0f, 0.0f, 0.0f, right_n);
  const int right1 = add_vertex(geometry, 555.0f, 555.0f, 0.0f, right_n);
  const int right2 = add_vertex(geometry, 555.0f, 555.0f, 555.0f, right_n);
  const int right3 = add_vertex(geometry, 555.0f, 0.0f, 555.0f, right_n);
  add_quad(geometry, kGreen, kObjectRightWall, right0, right1, right2, right3);
}

void add_default_materials(SceneConfig& config, float light_intensity) {
    config.materials = {
        {make_vec(0.725f, 0.71f, 0.68f), make_vec(0.0f, 0.0f, 0.0f), kMaterialDiffuse, 1.0f},
        {make_vec(0.63f, 0.065f, 0.05f), make_vec(0.0f, 0.0f, 0.0f), kMaterialDiffuse, 1.0f},
        {make_vec(0.14f, 0.45f, 0.091f), make_vec(0.0f, 0.0f, 0.0f), kMaterialDiffuse, 1.0f},
        {make_vec(1.0f, 1.0f, 1.0f), make_vec(light_intensity, light_intensity, light_intensity), kMaterialDiffuse, 1.0f},
        {make_vec(1.0f, 1.0f, 1.0f), make_vec(0.0f, 0.0f, 0.0f), kMaterialDielectric, 1.8f},
    };
}

void set_default_camera(SceneConfig& config) {
    config.camera_pos = make_vec(277.5f, 277.5f, -1600.0f);
    config.camera_lookat = make_vec(277.5f, 277.5f, 277.5f);
    config.camera_up = make_vec(0.0f, 1.0f, 0.0f);
    config.camera_fov_y = 19.5f * 3.1415926535f / 180.0f;
}

void add_default_light(SceneConfig& config, float light_intensity, float light_radius, unsigned int light_type, float reference_radius) {
    constexpr int kLight = 3;
  // Sphere lights are represented by both analytic sampling and a visible emissive mesh.
    if (light_type == kLightTypeSphere && light_radius > 1.0e-4f) {
        const float3 light_center = make_vec(277.5f, 540.0f, 277.5f);
        add_uv_sphere(config.geometry, light_center, light_radius, kLight, kObjectLight);
    }
    
    Light primary_light = {};
    primary_light.type = light_type;
    primary_light.position = make_vec(277.5f, 535.0f, 277.5f);
    primary_light.u = make_vec(0.0f, -1.0f, 0.0f); // Default direction
    primary_light.v = make_vec(cosf(45.0f * 3.1415926535f / 180.0f), 0.0f, 0.0f);
    primary_light.radius = (light_type == kLightTypePoint || light_type == kLightTypeSpot) ? 0.0f : light_radius;
    
    float emission_val = light_intensity;
    if (light_type == kLightTypePoint || light_type == kLightTypeSpot) {
        const float reference_area = 4.0f * 3.1415926535f * reference_radius * reference_radius;
        emission_val = light_intensity * reference_area * 0.25f;
    }
    primary_light.emission = make_vec(emission_val, emission_val, emission_val);
    config.lights.push_back(primary_light);
}
} // namespace

SceneConfig build_cornell_box_water(float light_intensity, float light_radius, unsigned int light_type, float reference_radius) {
    SceneConfig config;
    add_cornell_box_walls(config.geometry);
    add_default_materials(config, light_intensity);
    set_default_camera(config);
    add_default_light(config, light_intensity, light_radius, light_type, reference_radius);

    constexpr int kWhite = 0;
    constexpr int kGlass = 4;

    // ShortBox
    add_box(config.geometry, make_vec(375.0f, 0.0f, 355.0f), make_vec(465.0f, 90.0f, 445.0f), kWhite, kObjectShortBox);
    // Glass Sphere
    add_uv_sphere(config.geometry, make_vec(130.0f, 90.0f, 197.5f), 90.0f, kGlass, kObjectTallBox);

    // Water Mesh
    const float3 water_scale = make_vec(277.5f, 500.0f, 277.5f);
    const float3 water_offset = make_vec(277.5f, -300.0f, 277.5f);
    add_ply_mesh(config.geometry, "models/Mesh001.ply", kGlass, kObjectWater, water_scale, water_offset);

    // Imported caustic test mesh
    const float imported_mesh_scale = 60.0f;
    const float3 imported_mesh_scale_vec = make_vec(imported_mesh_scale, imported_mesh_scale, imported_mesh_scale);
    const float3 imported_mesh_offset = make_vec(644.0079f, -89.9528f, 24.4905f);
    add_ply_mesh(config.geometry, "models/mesh_00001.ply", kGlass, kObjectImportedGlass, imported_mesh_scale_vec, imported_mesh_offset);

    return config;
}

SceneConfig build_cornell_box_glass(float light_intensity, float light_radius, unsigned int light_type, float reference_radius) {
    SceneConfig config;
    add_cornell_box_walls(config.geometry);
    add_default_materials(config, light_intensity);
    set_default_camera(config);
  // Scene2 uses a compact, high-intensity sphere light inside the glass mesh.
    
    constexpr int kWhite = 0;
    constexpr int kGlass = 4;
    
    // Imported caustic test mesh at the center of which the light will be
    const float imported_mesh_scale = 60.0f;
    const float3 imported_mesh_scale_vec = make_vec(imported_mesh_scale, imported_mesh_scale, imported_mesh_scale);
    const float3 imported_mesh_offset = make_vec(644.0079f, -89.9528f, 24.4905f);
    add_ply_mesh(config.geometry, "models/mesh_00001.ply", kGlass, kObjectImportedGlass, imported_mesh_scale_vec, imported_mesh_offset);

    const float3 light_pos = make_vec(420.0f, 60.0f, 170.0f);
    const float sphere_radius = light_radius > 1.0e-4f ? fminf(light_radius, 1.0f) : 1.0f;

    const float reference_area = 4.0f * 3.1415926535f * reference_radius * reference_radius;
    const float emission_val = light_intensity * reference_area * 0.03f;

    // Add a visible emissive sphere so the light source can be seen directly.
    const int light_material = static_cast<int>(config.materials.size());
    config.materials.push_back(
      {make_vec(1.0f, 1.0f, 1.0f), make_vec(emission_val, emission_val, emission_val), kMaterialDiffuse, 1.0f});
    add_uv_sphere(config.geometry, light_pos, sphere_radius, light_material, kObjectLight);

    // Sphere light used by direct lighting / photon emission.
    Light mesh_light = {};
    mesh_light.type = kLightTypeSphere;
    mesh_light.position = light_pos;
    mesh_light.radius = sphere_radius;
    mesh_light.emission = make_vec(emission_val, emission_val, emission_val);
    config.lights.push_back(mesh_light);

    // Position TallBox in the far back-left corner as requested for verification
    add_box(config.geometry, make_vec(50.0f, 0.0f, 400.0f), make_vec(210.0f, 165.0f, 550.0f), kWhite, kObjectTallBox);

    return config;
}