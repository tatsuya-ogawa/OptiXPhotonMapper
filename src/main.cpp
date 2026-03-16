#include "shared_defs.h"
#include "photon_helpers.h"
#include "scene.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) Record {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

using RaygenRecord = Record<int>;
using MissRecord = Record<int>;
using SceneHitgroupRecord = Record<SceneHitGroupData>;
using PhotonHitgroupRecord = Record<PhotonHitGroupData>;

struct GasBuildResult {
  OptixTraversableHandle handle = 0;
  CUdeviceptr buffer = 0;
};

static void check_cuda(cudaError_t result, const char* call) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(result));
  }
}

static void check_optix(OptixResult result, const char* call) {
  if (result != OPTIX_SUCCESS) {
    throw std::runtime_error(std::string(call) + " failed with code " + std::to_string(result));
  }
}

static void save_ppm(const std::string& path,
                     const std::vector<float4>& pixels,
                     unsigned int width,
                     unsigned int height) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open output file: " + path);
  }
  out << "P6\n" << width << " " << height << "\n255\n";
  for (const auto& px : pixels) {
    const float r = std::sqrt(std::fmax(px.x, 0.0f));
    const float g = std::sqrt(std::fmax(px.y, 0.0f));
    const float b = std::sqrt(std::fmax(px.z, 0.0f));
    out.put(static_cast<char>(std::fmin(r, 1.0f) * 255.99f));
    out.put(static_cast<char>(std::fmin(g, 1.0f) * 255.99f));
    out.put(static_cast<char>(std::fmin(b, 1.0f) * 255.99f));
  }
}

static OptixImage2D make_optix_image(CUdeviceptr data, unsigned int width, unsigned int height) {
  OptixImage2D image = {};
  image.data = data;
  image.width = width;
  image.height = height;
  image.rowStrideInBytes = width * sizeof(float4);
  image.pixelStrideInBytes = sizeof(float4);
  image.format = OPTIX_PIXEL_FORMAT_FLOAT4;
  return image;
}

static void denoise_beauty(OptixDeviceContext context,
                           unsigned int width,
                           unsigned int height,
                           CUdeviceptr input,
                           CUdeviceptr output) {
  OptixDenoiserOptions options = {};
  options.guideAlbedo = 0;
  options.guideNormal = 0;
  options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

  OptixDenoiser denoiser = nullptr;
  check_optix(optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &denoiser),
              "optixDenoiserCreate");

  OptixDenoiserSizes sizes = {};
  check_optix(optixDenoiserComputeMemoryResources(denoiser, width, height, &sizes),
              "optixDenoiserComputeMemoryResources");

  CUdeviceptr denoiser_state = 0;
  CUdeviceptr scratch = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&denoiser_state), sizes.stateSizeInBytes),
             "cudaMalloc(denoiser_state)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&scratch), sizes.withoutOverlapScratchSizeInBytes),
             "cudaMalloc(denoiser_scratch)");

  check_optix(optixDenoiserSetup(denoiser,
                                 0,
                                 width,
                                 height,
                                 denoiser_state,
                                 sizes.stateSizeInBytes,
                                 scratch,
                                 sizes.withoutOverlapScratchSizeInBytes),
              "optixDenoiserSetup");

  OptixDenoiserLayer layer = {};
  layer.input = make_optix_image(input, width, height);
  layer.output = make_optix_image(output, width, height);
  layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
  OptixDenoiserGuideLayer guide = {};

  OptixDenoiserParams params = {};
  params.hdrIntensity = 0;
  params.blendFactor = 0.0f;
  params.hdrAverageColor = 0;
  params.temporalModeUsePreviousLayers = 0;

  check_optix(optixDenoiserInvoke(denoiser,
                                  0,
                                  &params,
                                  denoiser_state,
                                  sizes.stateSizeInBytes,
                                  &guide,
                                  &layer,
                                  1,
                                  0,
                                  0,
                                  scratch,
                                  sizes.withoutOverlapScratchSizeInBytes),
              "optixDenoiserInvoke");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(denoiser)");

  check_cuda(cudaFree(reinterpret_cast<void*>(scratch)), "cudaFree(denoiser_scratch)");
  check_cuda(cudaFree(reinterpret_cast<void*>(denoiser_state)), "cudaFree(denoiser_state)");
  check_optix(optixDenoiserDestroy(denoiser), "optixDenoiserDestroy");
}

static std::string load_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
  std::cerr << "[" << level << "][" << tag << "] " << message << "\n";
}

static GasBuildResult build_triangle_gas(OptixDeviceContext context,
                                         CUdeviceptr vertex_buffer,
                                         unsigned int vertex_count,
                                         CUdeviceptr index_buffer,
                                         unsigned int index_count) {
  OptixBuildInput build_input = {};
  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
  build_input.triangleArray.vertexBuffers = &vertex_buffer;
  build_input.triangleArray.numVertices = vertex_count;
  build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
  build_input.triangleArray.indexBuffer = index_buffer;
  build_input.triangleArray.numIndexTriplets = index_count;
  build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
  build_input.triangleArray.flags = flags;
  build_input.triangleArray.numSbtRecords = 1;

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes sizes = {};
  check_optix(optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &sizes),
              "optixAccelComputeMemoryUsage(triangle)");

  CUdeviceptr temp = 0;
  CUdeviceptr output = 0;
  CUdeviceptr compact_size = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&temp), sizes.tempSizeInBytes), "cudaMalloc(tri_temp)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&output), sizes.outputSizeInBytes),
             "cudaMalloc(tri_output)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&compact_size), sizeof(uint64_t)),
             "cudaMalloc(tri_compact_size)");

  OptixAccelEmitDesc emit = {};
  emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit.result = compact_size;

  GasBuildResult result;
  check_optix(optixAccelBuild(context,
                              0,
                              &accel_options,
                              &build_input,
                              1,
                              temp,
                              sizes.tempSizeInBytes,
                              output,
                              sizes.outputSizeInBytes,
                              &result.handle,
                              &emit,
                              1),
              "optixAccelBuild(triangle)");
  check_cuda(cudaFree(reinterpret_cast<void*>(temp)), "cudaFree(tri_temp)");

  uint64_t compacted_size = 0;
  check_cuda(cudaMemcpy(&compacted_size,
                        reinterpret_cast<void*>(compact_size),
                        sizeof(uint64_t),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(tri_compact_size)");
  check_cuda(cudaFree(reinterpret_cast<void*>(compact_size)), "cudaFree(tri_compact_size)");

  result.buffer = output;
  if (compacted_size > 0 && compacted_size < sizes.outputSizeInBytes) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&result.buffer), compacted_size),
               "cudaMalloc(tri_compacted)");
    check_optix(optixAccelCompact(context, 0, result.handle, result.buffer, compacted_size, &result.handle),
                "optixAccelCompact(triangle)");
    check_cuda(cudaFree(reinterpret_cast<void*>(output)), "cudaFree(tri_uncompacted)");
  }
  return result;
}

static GasBuildResult build_photon_gas(OptixDeviceContext context,
                                       CUdeviceptr aabb_buffer,
                                       unsigned int photon_count) {
  OptixBuildInput build_input = {};
  build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
  build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
  build_input.customPrimitiveArray.numPrimitives = photon_count;
  build_input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
  build_input.customPrimitiveArray.flags = flags;
  build_input.customPrimitiveArray.numSbtRecords = 1;

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes sizes = {};
  check_optix(optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &sizes),
              "optixAccelComputeMemoryUsage(photon)");

  CUdeviceptr temp = 0;
  CUdeviceptr output = 0;
  CUdeviceptr compact_size = 0;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&temp), sizes.tempSizeInBytes), "cudaMalloc(photon_temp)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&output), sizes.outputSizeInBytes),
             "cudaMalloc(photon_output)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&compact_size), sizeof(uint64_t)),
             "cudaMalloc(photon_compact_size)");

  OptixAccelEmitDesc emit = {};
  emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit.result = compact_size;

  GasBuildResult result;
  check_optix(optixAccelBuild(context,
                              0,
                              &accel_options,
                              &build_input,
                              1,
                              temp,
                              sizes.tempSizeInBytes,
                              output,
                              sizes.outputSizeInBytes,
                              &result.handle,
                              &emit,
                              1),
              "optixAccelBuild(photon)");
  check_cuda(cudaFree(reinterpret_cast<void*>(temp)), "cudaFree(photon_temp)");

  uint64_t compacted_size = 0;
  check_cuda(cudaMemcpy(&compacted_size,
                        reinterpret_cast<void*>(compact_size),
                        sizeof(uint64_t),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(photon_compact_size)");
  check_cuda(cudaFree(reinterpret_cast<void*>(compact_size)), "cudaFree(photon_compact_size)");

  result.buffer = output;
  if (compacted_size > 0 && compacted_size < sizes.outputSizeInBytes) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&result.buffer), compacted_size),
               "cudaMalloc(photon_compacted)");
    check_optix(optixAccelCompact(context, 0, result.handle, result.buffer, compacted_size, &result.handle),
                "optixAccelCompact(photon)");
    check_cuda(cudaFree(reinterpret_cast<void*>(output)), "cudaFree(photon_uncompacted)");
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string output_filename = argc > 1 ? argv[1] : "cornell.ppm";
    
    // Output directory management
    std::string env_output_dir = "";
    if (const char* env_dir = std::getenv("OUTPUT_DIR")) {
        env_output_dir = env_dir;
    }
    
    std::filesystem::path final_output_path;
    std::filesystem::path given_path(output_filename);
    
    if (given_path.is_absolute()) {
        final_output_path = given_path;
    } else {
        std::filesystem::path base_dir = env_output_dir.empty() ? "images" : env_output_dir;
        final_output_path = base_dir / given_path;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::path parent_dir = final_output_path.parent_path();
    if (!parent_dir.empty() && !std::filesystem::exists(parent_dir)) {
        std::filesystem::create_directories(parent_dir);
    }
    
    const std::string output_path = final_output_path.string();
    const unsigned int width = argc > 2 ? static_cast<unsigned int>(std::stoi(argv[2])) : 800;
    const unsigned int height = argc > 3 ? static_cast<unsigned int>(std::stoi(argv[3])) : 800;
    const unsigned int samples = argc > 4 ? static_cast<unsigned int>(std::stoi(argv[4])) : 8;
    const unsigned int photon_launch_count =
        argc > 5 ? static_cast<unsigned int>(std::stoi(argv[5])) : 60000;
    const unsigned int render_mode =
        argc > 6 ? static_cast<unsigned int>(std::stoi(argv[6])) : kRenderModeBeauty;
    const float photon_radius = argc > 7 ? std::stof(argv[7]) : 35.0f;
    const float indirect_scale = argc > 8 ? std::stof(argv[8]) : 1.0f;
    const unsigned int photon_max_bounces =
        argc > 9 ? static_cast<unsigned int>(std::stoi(argv[9])) : 10;
    const float photon_normal_reject_cos = argc > 10 ? std::stof(argv[10]) : 0.9f;
    const unsigned int ppm_iterations =
        argc > 11 ? static_cast<unsigned int>(std::stoi(argv[11])) : 1;
    const float ppm_alpha = argc > 12 ? std::stof(argv[12]) : 0.7f;
    const bool debug_stats = argc > 13 ? std::stoi(argv[13]) != 0 : false;
    const unsigned int photon_eval_mode =
        argc > 14 ? static_cast<unsigned int>(std::stoi(argv[14])) : kPhotonEvalStochastic;
    const unsigned int direct_light_samples =
        argc > 15 ? static_cast<unsigned int>(std::stoi(argv[15])) : 1;
    const bool enable_denoiser = argc > 16 ? std::stoi(argv[16]) != 0 : false;
    const float light_radius = argc > 17 ? std::stof(argv[17]) : 20.0f;
    const unsigned int light_type = []() {
        if (const char* env_light = std::getenv("LIGHT_TYPE")) {
            return static_cast<unsigned int>(std::stoi(env_light));
        }
        return kLightTypeSphere; // Default
    }();
    const float caustic_photon_radius = std::fmax(1.0f, photon_radius * 0.25f);
    const float global_photon_rejection = argc > 18 ? std::stof(argv[18]) : 0.3f;
    const bool use_multi_diffuse_caustic_map = argc > 19 ? std::stoi(argv[19]) != 0 : false;
    
    // Check for LIGHT_INTENSITY environment variable
    float default_light_intensity = 50.0f;
    if (const char* env_intensity = std::getenv("LIGHT_INTENSITY")) {
        try {
            default_light_intensity = std::stof(env_intensity);
        } catch (...) {
            // Ignore invalid environment variable values
        }
    }
    const float light_intensity = argc > 20 ? std::stof(argv[20]) : default_light_intensity;

    std::string scene_name = "water";
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--scene") {
            scene_name = argv[i+1];
        }
    }

    const float point_power_reference_radius = []() {
      if (const char* env_ref_radius = std::getenv("POINT_POWER_REFERENCE_RADIUS")) {
        try {
          return std::fmax(0.0f, std::stof(env_ref_radius));
        } catch (...) {
        }
      }
      return 20.0f;
    }();
    const bool use_photon_mapping = photon_launch_count > 0;

    check_optix(optixInit(), "optixInit");
    check_cuda(cudaFree(nullptr), "cudaFree");

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = context_log_cb;
    options.logCallbackLevel = 4;
    OptixDeviceContext context = nullptr;
    check_optix(optixDeviceContextCreate(nullptr, &options, &context), "optixDeviceContextCreate");

    SceneConfig scene_config;
    std::string env_scene_val = "";
    if (const char* env_scene = std::getenv("SCENE_NAME")) {
        env_scene_val = env_scene;
    }
    if (!env_scene_val.empty()) {
        scene_name = env_scene_val;
    }
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--scene") {
            scene_name = argv[i+1];
        }
    }

    if (scene_name == "water") {
        scene_config = build_cornell_box_water(light_intensity, light_radius, light_type, point_power_reference_radius);
    } else if (scene_name == "glass") {
        scene_config = build_cornell_box_glass(light_intensity, light_radius, light_type, point_power_reference_radius);
    } else {
        std::cerr << "Unknown scene: " << scene_name << ". Defaulting to 'water'.\n";
        scene_config = build_cornell_box_water(light_intensity, light_radius, light_type, point_power_reference_radius);
    }

    // Restore Spot light configuration if applicable
    if (light_type == kLightTypeSpot && !scene_config.lights.empty()) {
        Light& primary_light = scene_config.lights[0];
        if (const char* env_dir = std::getenv("SPOT_DIRECTION")) {
            float x, y, z;
            if (sscanf(env_dir, "%f,%f,%f", &x, &y, &z) == 3) {
                primary_light.u = make_vec(x, y, z);
            }
        }
        if (const char* env_cutoff = std::getenv("SPOT_CUTOFF")) {
            primary_light.v.x = cosf(std::stof(env_cutoff) * 3.1415926535f / 180.0f);
        }
        if (const char* env_exp = std::getenv("SPOT_EXPONENT")) {
            primary_light.v.y = std::stof(env_exp);
        }
    } else if (light_type == kLightTypeQuad && !scene_config.lights.empty()) {
        Light& primary_light = scene_config.lights[0];
        primary_light.u = make_vec(130.0f, 0.0f, 0.0f);
        primary_light.v = make_vec(0.0f, 0.0f, 130.0f);
    }

    GeometryData& geometry = scene_config.geometry;
    std::vector<Material>& materials = scene_config.materials;

    float3* d_vertices = nullptr;
    float3* d_normals = nullptr;
    uint3* d_indices = nullptr;
    int* d_material_indices = nullptr;
    int* d_object_indices = nullptr;
    Material* d_materials = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_vertices), geometry.vertices.size() * sizeof(float3)),
               "cudaMalloc(vertices)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_normals), geometry.normals.size() * sizeof(float3)),
               "cudaMalloc(normals)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_indices), geometry.indices.size() * sizeof(uint3)),
               "cudaMalloc(indices)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_material_indices),
                          geometry.material_indices.size() * sizeof(int)),
               "cudaMalloc(material_indices)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_object_indices),
                          geometry.object_indices.size() * sizeof(int)),
               "cudaMalloc(object_indices)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_materials), materials.size() * sizeof(Material)),
               "cudaMalloc(materials)");

    check_cuda(cudaMemcpy(d_vertices,
                          geometry.vertices.data(),
                          geometry.vertices.size() * sizeof(float3),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(vertices)");
    check_cuda(cudaMemcpy(d_normals,
                          geometry.normals.data(),
                          geometry.normals.size() * sizeof(float3),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(normals)");
    check_cuda(cudaMemcpy(d_indices,
                          geometry.indices.data(),
                          geometry.indices.size() * sizeof(uint3),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(indices)");
    check_cuda(cudaMemcpy(d_material_indices,
                          geometry.material_indices.data(),
                          geometry.material_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(material_indices)");
    check_cuda(cudaMemcpy(d_object_indices,
                          geometry.object_indices.data(),
                          geometry.object_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(object_indices)");
    check_cuda(cudaMemcpy(d_materials,
                          materials.data(),
                          materials.size() * sizeof(Material),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(materials)");

    GasBuildResult scene_gas = build_triangle_gas(context,
                                                  reinterpret_cast<CUdeviceptr>(d_vertices),
                                                  static_cast<unsigned int>(geometry.vertices.size()),
                                                  reinterpret_cast<CUdeviceptr>(d_indices),
                                                  static_cast<unsigned int>(geometry.indices.size()));

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    const char* env_ptx_path = std::getenv("PTX_PATH");
    std::string ptx_path = env_ptx_path ? std::string(env_ptx_path) : "build/renderer.ptx";
    std::string ptx = load_file(ptx_path);
    char log[2048];
    size_t log_size = sizeof(log);
    OptixModule module = nullptr;
    check_optix(optixModuleCreate(context,
                                  &module_compile_options,
                                  &pipeline_compile_options,
                                  ptx.c_str(),
                                  ptx.size(),
                                  log,
                                  &log_size,
                                  &module),
                "optixModuleCreate");

    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc raygen_descs[2] = {};
    raygen_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_descs[0].raygen.module = module;
    raygen_descs[0].raygen.entryFunctionName = "__raygen__render";
    raygen_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_descs[1].raygen.module = module;
    raygen_descs[1].raygen.entryFunctionName = "__raygen__emit_photons";

    OptixProgramGroupDesc miss_descs[kRayTypeCount] = {};
    const char* miss_names[kRayTypeCount] = {
        "__miss__radiance",
        "__miss__shadow",
        "__miss__photon_trace",
        "__miss__photon_gather",
        "__miss__photon_gather",
    };
    for (unsigned int i = 0; i < kRayTypeCount; ++i) {
      miss_descs[i].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      miss_descs[i].miss.module = module;
      miss_descs[i].miss.entryFunctionName = miss_names[i];
    }

    OptixProgramGroupDesc scene_radiance_desc = {};
    scene_radiance_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    scene_radiance_desc.hitgroup.moduleCH = module;
    scene_radiance_desc.hitgroup.entryFunctionNameCH = "__closesthit__scene_radiance";

    OptixProgramGroupDesc scene_shadow_desc = {};
    scene_shadow_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    scene_shadow_desc.hitgroup.moduleAH = module;
    scene_shadow_desc.hitgroup.entryFunctionNameAH = "__anyhit__scene_shadow";

    OptixProgramGroupDesc scene_photon_desc = {};
    scene_photon_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    scene_photon_desc.hitgroup.moduleCH = module;
    scene_photon_desc.hitgroup.entryFunctionNameCH = "__closesthit__scene_photon";

    OptixProgramGroupDesc photon_gather_desc = {};
    photon_gather_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    photon_gather_desc.hitgroup.moduleIS = module;
    photon_gather_desc.hitgroup.entryFunctionNameIS = "__intersection__photon";
    photon_gather_desc.hitgroup.moduleAH = module;
    photon_gather_desc.hitgroup.entryFunctionNameAH = "__anyhit__photon_gather";

    OptixProgramGroup raygen_pgs[2] = {};
    OptixProgramGroup miss_pgs[kRayTypeCount] = {};
    OptixProgramGroup scene_radiance_pg = nullptr;
    OptixProgramGroup scene_shadow_pg = nullptr;
    OptixProgramGroup scene_photon_pg = nullptr;
    OptixProgramGroup photon_gather_pg = nullptr;

    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        raygen_descs,
                                        2,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        raygen_pgs),
                "optixProgramGroupCreate(raygen)");
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        miss_descs,
                                        kRayTypeCount,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        miss_pgs),
                "optixProgramGroupCreate(miss)");
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &scene_radiance_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &scene_radiance_pg),
                "optixProgramGroupCreate(scene_radiance)");
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &scene_shadow_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &scene_shadow_pg),
                "optixProgramGroupCreate(scene_shadow)");
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &scene_photon_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &scene_photon_pg),
                "optixProgramGroupCreate(scene_photon)");
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(context,
                                        &photon_gather_desc,
                                        1,
                                        &pg_options,
                                        log,
                                        &log_size,
                                        &photon_gather_pg),
                "optixProgramGroupCreate(photon_gather)");

    std::array<OptixProgramGroup, 11> program_groups = {
        raygen_pgs[0], raygen_pgs[1], miss_pgs[0], miss_pgs[1], miss_pgs[2],
        miss_pgs[3], miss_pgs[4], scene_radiance_pg, scene_shadow_pg, scene_photon_pg,
        photon_gather_pg};

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 4;

    OptixPipeline pipeline = nullptr;
    log_size = sizeof(log);
    check_optix(optixPipelineCreate(context,
                                    &pipeline_compile_options,
                                    &link_options,
                                    program_groups.data(),
                                    static_cast<unsigned int>(program_groups.size()),
                                    log,
                                    &log_size,
                                    &pipeline),
                "optixPipelineCreate");

    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup pg : program_groups) {
      check_optix(optixUtilAccumulateStackSizes(pg, &stack_sizes, pipeline),
                  "optixUtilAccumulateStackSizes");
    }
    uint32_t dc_stack_from_traversal = 0;
    uint32_t dc_stack_from_state = 0;
    uint32_t continuation_stack = 0;
    check_optix(optixUtilComputeStackSizes(&stack_sizes,
                                           link_options.maxTraceDepth,
                                           0,
                                           0,
                                           &dc_stack_from_traversal,
                                           &dc_stack_from_state,
                                           &continuation_stack),
                "optixUtilComputeStackSizes");
    check_optix(optixPipelineSetStackSize(pipeline,
                                          dc_stack_from_traversal,
                                          dc_stack_from_state,
                                          continuation_stack,
                                          1),
                "optixPipelineSetStackSize");

    RaygenRecord raygen_records[2] = {};
    check_optix(optixSbtRecordPackHeader(raygen_pgs[0], &raygen_records[0]),
                "optixSbtRecordPackHeader(raygen_render)");
    check_optix(optixSbtRecordPackHeader(raygen_pgs[1], &raygen_records[1]),
                "optixSbtRecordPackHeader(raygen_photon)");

    MissRecord miss_records[kRayTypeCount] = {};
    for (unsigned int i = 0; i < kRayTypeCount; ++i) {
      check_optix(optixSbtRecordPackHeader(miss_pgs[i], &miss_records[i]),
                  "optixSbtRecordPackHeader(miss)");
    }

    SceneHitgroupRecord scene_record = {};
    scene_record.data.vertices = d_vertices;
    scene_record.data.normals = d_normals;
    scene_record.data.indices = d_indices;
    scene_record.data.material_indices = d_material_indices;
    scene_record.data.object_indices = d_object_indices;
    scene_record.data.materials = d_materials;

    PhotonHitgroupRecord photon_record = {};

    std::array<unsigned char, sizeof(SceneHitgroupRecord)> scene_bytes = {};
    std::array<unsigned char, sizeof(PhotonHitgroupRecord)> photon_bytes = {};

    SceneHitgroupRecord scene_radiance_record = scene_record;
    SceneHitgroupRecord scene_shadow_record = scene_record;
    SceneHitgroupRecord scene_photon_record = scene_record;
    PhotonHitgroupRecord caustic_photon_gather_record = photon_record;
    caustic_photon_gather_record.data.photons = nullptr;
    caustic_photon_gather_record.data.radius = photon_radius;
    PhotonHitgroupRecord global_photon_gather_record = photon_record;
    global_photon_gather_record.data.photons = nullptr;
    global_photon_gather_record.data.radius = photon_radius;

    check_optix(optixSbtRecordPackHeader(scene_radiance_pg, &scene_radiance_record),
                "optixSbtRecordPackHeader(scene_radiance)");
    check_optix(optixSbtRecordPackHeader(scene_shadow_pg, &scene_shadow_record),
                "optixSbtRecordPackHeader(scene_shadow)");
    check_optix(optixSbtRecordPackHeader(scene_photon_pg, &scene_photon_record),
                "optixSbtRecordPackHeader(scene_photon)");
    check_optix(optixSbtRecordPackHeader(photon_gather_pg, &caustic_photon_gather_record),
                "optixSbtRecordPackHeader(caustic_photon_gather)");
    check_optix(optixSbtRecordPackHeader(photon_gather_pg, &global_photon_gather_record),
                "optixSbtRecordPackHeader(global_photon_gather)");

    std::vector<unsigned char> hitgroup_blob(sizeof(SceneHitgroupRecord) * kRayTypeCount);
    auto copy_record = [&](unsigned int slot, const void* record, size_t size) {
      std::memcpy(hitgroup_blob.data() + slot * sizeof(SceneHitgroupRecord), record, size);
    };

    copy_record(kRadianceRayType, &scene_radiance_record, sizeof(SceneHitgroupRecord));
    copy_record(kShadowRayType, &scene_shadow_record, sizeof(SceneHitgroupRecord));
    copy_record(kPhotonTraceRayType, &scene_photon_record, sizeof(SceneHitgroupRecord));
    copy_record(kPhotonGatherCausticRayType,
                &caustic_photon_gather_record,
                sizeof(PhotonHitgroupRecord));
    copy_record(kPhotonGatherGlobalRayType,
                &global_photon_gather_record,
                sizeof(PhotonHitgroupRecord));

    CUdeviceptr d_raygen_records = 0;
    CUdeviceptr d_miss_records = 0;
    CUdeviceptr d_hitgroup_records = 0;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_raygen_records), sizeof(raygen_records)),
               "cudaMalloc(raygen_records)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), sizeof(miss_records)),
               "cudaMalloc(miss_records)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), hitgroup_blob.size()),
               "cudaMalloc(hitgroup_records)");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(d_raygen_records),
                          raygen_records,
                          sizeof(raygen_records),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(raygen_records)");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(d_miss_records),
                          miss_records,
                          sizeof(miss_records),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(miss_records)");
    check_cuda(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records),
                          hitgroup_blob.data(),
                          hitgroup_blob.size(),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(hitgroup_records)");

    OptixShaderBindingTable render_sbt = {};
    render_sbt.raygenRecord = d_raygen_records;
    render_sbt.missRecordBase = d_miss_records;
    render_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    render_sbt.missRecordCount = kRayTypeCount;
    render_sbt.hitgroupRecordBase = d_hitgroup_records;
    render_sbt.hitgroupRecordStrideInBytes = sizeof(SceneHitgroupRecord);
    render_sbt.hitgroupRecordCount = kRayTypeCount;

    OptixShaderBindingTable photon_sbt = render_sbt;
    photon_sbt.raygenRecord = d_raygen_records + sizeof(RaygenRecord);

    const unsigned int max_photons =
        use_photon_mapping ? photon_launch_count * photon_max_bounces : 0;
    Photon* d_global_photons = nullptr;
    Photon* d_caustic_photons = nullptr;
    unsigned int* d_global_photon_count = nullptr;
    unsigned int* d_caustic_photon_count = nullptr;
    OptixAabb* d_global_photon_aabbs = nullptr;
    OptixAabb* d_caustic_photon_aabbs = nullptr;
    if (use_photon_mapping) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_global_photons), max_photons * sizeof(Photon)),
                 "cudaMalloc(global_photons)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_caustic_photons), max_photons * sizeof(Photon)),
                 "cudaMalloc(caustic_photons)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_global_photon_count), sizeof(unsigned int)),
                 "cudaMalloc(global_photon_count)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_caustic_photon_count), sizeof(unsigned int)),
                 "cudaMalloc(caustic_photon_count)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_global_photon_aabbs),
                            max_photons * sizeof(OptixAabb)),
                 "cudaMalloc(global_photon_aabbs)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_caustic_photon_aabbs),
                            max_photons * sizeof(OptixAabb)),
                 "cudaMalloc(caustic_photon_aabbs)");
      check_cuda(cudaMemset(d_global_photon_count, 0, sizeof(unsigned int)),
                 "cudaMemset(global_photon_count)");
      check_cuda(cudaMemset(d_caustic_photon_count, 0, sizeof(unsigned int)),
                 "cudaMemset(caustic_photon_count)");
    }

    float4* d_output = nullptr;
    float4* d_accumulated_output = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output), width * height * sizeof(float4)),
               "cudaMalloc(output)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_accumulated_output),
                          width * height * sizeof(float4)),
               "cudaMalloc(accumulated_output)");
    clear_float4_buffer(d_accumulated_output, width * height);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(clear_accumulated_output)");

    const float aspect = static_cast<float>(width) / static_cast<float>(height);

    Params params = {};
    params.image = d_output;
    params.width = width;
    params.height = height;
    params.samples_per_pixel = samples;
    params.iteration_index = 0;
    params.photon_launch_count = photon_launch_count;
    params.photon_max_bounces = photon_max_bounces;
    params.max_photons = max_photons;
    params.direct_light_samples = direct_light_samples;
    params.photon_eval_mode = photon_eval_mode;
    params.render_mode = render_mode;
    params.use_multi_diffuse_caustic_map = use_multi_diffuse_caustic_map ? 1u : 0u;
    params.photon_normal_reject_cos = photon_normal_reject_cos;
    params.global_photon_rejection = global_photon_rejection;
    params.global_photon_gather_scale = indirect_scale / (3.1415926535f * photon_radius * photon_radius);
    params.caustic_photon_gather_scale =
        indirect_scale / (3.1415926535f * caustic_photon_radius * caustic_photon_radius);
    params.global_photons = d_global_photons;
    params.caustic_photons = d_caustic_photons;
    params.global_photon_count = d_global_photon_count;
    params.caustic_photon_count = d_caustic_photon_count;
    params.scene_handle = scene_gas.handle;
    params.caustic_photon_handle = 0;

    std::vector<Light>& h_lights = scene_config.lights;

    params.light_count = static_cast<unsigned int>(h_lights.size());
    Light* d_lights = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_lights), h_lights.size() * sizeof(Light)), "cudaMalloc(lights)");
    check_cuda(cudaMemcpy(d_lights, h_lights.data(), h_lights.size() * sizeof(Light), cudaMemcpyHostToDevice), "cudaMemcpy(lights)");
    params.lights = d_lights;

    params.camera_position = scene_config.camera_pos;
    const float3 forward = normalize3(scene_config.camera_lookat - scene_config.camera_pos);
    const float3 right = normalize3(cross3(forward, scene_config.camera_up));
    const float3 camera_up = normalize3(cross3(right, forward));
    
    const float cam_dist = length3(scene_config.camera_lookat - scene_config.camera_pos);
    const float half_height = std::tan(0.5f * scene_config.camera_fov_y);
    const float half_width = aspect * half_height;

    params.camera_u = right * half_width * cam_dist;
    params.camera_v = camera_up * half_height * cam_dist;
    params.camera_w = forward * cam_dist;

    Params* d_params = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)), "cudaMalloc(params)");
    check_cuda(cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice), "cudaMemcpy(params)");

    float current_global_radius = photon_radius;
    float current_caustic_radius = caustic_photon_radius;
    unsigned int last_photon_count = 0;

    for (unsigned int iteration = 0; iteration < ppm_iterations; ++iteration) {
      params.iteration_index = iteration;
      params.global_photon_handle = 0;
      params.caustic_photon_handle = 0;
      params.global_photon_gather_scale =
          indirect_scale / (3.1415926535f * current_global_radius * current_global_radius);
      params.caustic_photon_gather_scale =
          indirect_scale / (3.1415926535f * current_caustic_radius * current_caustic_radius);

      GasBuildResult global_photon_gas;
      GasBuildResult caustic_photon_gas;
      unsigned int global_photon_count = 0;
      unsigned int caustic_photon_count = 0;
      if (use_photon_mapping) {
        check_cuda(cudaMemset(d_global_photon_count, 0, sizeof(unsigned int)),
                   "cudaMemset(global_photon_count_loop)");
        check_cuda(cudaMemset(d_caustic_photon_count, 0, sizeof(unsigned int)),
                   "cudaMemset(caustic_photon_count_loop)");
        check_cuda(cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice),
                   "cudaMemcpy(params_photon)");
        check_optix(optixLaunch(pipeline,
                                0,
                                reinterpret_cast<CUdeviceptr>(d_params),
                                sizeof(Params),
                                &photon_sbt,
                                photon_launch_count,
                                1,
                                1),
                    "optixLaunch(photon)");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(photon)");

        check_cuda(cudaMemcpy(&global_photon_count,
                              d_global_photon_count,
                              sizeof(unsigned int),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(global_photon_count)");
        check_cuda(cudaMemcpy(&caustic_photon_count,
                              d_caustic_photon_count,
                              sizeof(unsigned int),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(caustic_photon_count)");
        if (global_photon_count > max_photons) {
          global_photon_count = max_photons;
        }
        if (caustic_photon_count > max_photons) {
          caustic_photon_count = max_photons;
        }
        last_photon_count = global_photon_count + caustic_photon_count;

        if (debug_stats || ppm_iterations > 1) {
          std::cout << "Iteration " << (iteration + 1) << "/" << ppm_iterations
                    << ": launched=" << photon_launch_count
                    << ", global=" << global_photon_count
                    << ", caustic=" << caustic_photon_count
                    << ", global_radius=" << current_global_radius
                    << ", caustic_radius=" << current_caustic_radius
                    << ", direct_light_samples=" << direct_light_samples
                    << ", indirect_scale=" << indirect_scale
                    << ", global_rejection=" << global_photon_rejection
                    << ", max_bounces=" << photon_max_bounces
                    << ", normal_reject_cos=" << photon_normal_reject_cos
                    << ", multi_diffuse_caustic=" << (use_multi_diffuse_caustic_map ? 1 : 0)
                    << ", alpha=" << ppm_alpha
                    << ", eval_mode=" << (photon_eval_mode == kPhotonEvalFull ? "full" : "stochastic")
                    << "\n";
        }

        if (debug_stats && iteration == 0 && (global_photon_count > 0 || caustic_photon_count > 0)) {
          const unsigned int preview_count = global_photon_count < 6 ? global_photon_count : 6;
          std::vector<Photon> preview(preview_count);
          if (preview_count > 0) {
            check_cuda(cudaMemcpy(preview.data(),
                                  d_global_photons,
                                  preview_count * sizeof(Photon),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(global_photon_preview)");
          }
          for (unsigned int i = 0; i < preview_count; ++i) {
            const Photon& p = preview[i];
            std::cout << "photon[" << i << "] pos=(" << p.position.x << ", " << p.position.y << ", "
                      << p.position.z << ") flux=(" << p.flux.x << ", " << p.flux.y << ", "
                      << p.flux.z << ") object_id=" << p.object_id << "\n";
          }

          std::array<unsigned int, kObjectCount> global_object_counts = {};
          std::array<unsigned int, kObjectCount> caustic_object_counts = {};
          if (global_photon_count > 0) {
            std::vector<Photon> global_photons(global_photon_count);
            check_cuda(cudaMemcpy(global_photons.data(),
                                  d_global_photons,
                                  global_photon_count * sizeof(Photon),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(global_photon_all)");
            for (const Photon& photon : global_photons) {
              if (photon.object_id < global_object_counts.size()) {
                ++global_object_counts[photon.object_id];
              }
            }
          }
          if (caustic_photon_count > 0) {
            std::vector<Photon> caustic_photons(caustic_photon_count);
            check_cuda(cudaMemcpy(caustic_photons.data(),
                                  d_caustic_photons,
                                  caustic_photon_count * sizeof(Photon),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(caustic_photon_all)");
            for (const Photon& photon : caustic_photons) {
              if (photon.object_id < caustic_object_counts.size()) {
                ++caustic_object_counts[photon.object_id];
              }
            }
          }
          static const char* kObjectNames[kObjectCount] = {
              "floor", "ceiling", "back_wall", "left_wall",
              "right_wall", "light", "short_box", "tall_box", "water",
              "imported_glass"};
          for (unsigned int i = 0; i < global_object_counts.size(); ++i) {
            std::cout << "global_hits[" << kObjectNames[i] << "]=" << global_object_counts[i] << "\n";
          }
          for (unsigned int i = 0; i < caustic_object_counts.size(); ++i) {
            std::cout << "caustic_hits[" << kObjectNames[i] << "]=" << caustic_object_counts[i] << "\n";
          }
        }

        if (global_photon_count > 0) {
          build_photon_aabbs(d_global_photons,
                             d_global_photon_aabbs,
                             global_photon_count,
                             current_global_radius);
          check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(global_photon_aabbs)");
          global_photon_gas = build_photon_gas(context,
                                               reinterpret_cast<CUdeviceptr>(d_global_photon_aabbs),
                                               global_photon_count);
          global_photon_gather_record.data.photons = d_global_photons;
          global_photon_gather_record.data.radius = current_global_radius;
          check_optix(optixSbtRecordPackHeader(photon_gather_pg, &global_photon_gather_record),
                      "optixSbtRecordPackHeader(global_photon_gather_refresh)");
          check_cuda(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records +
                                                        kPhotonGatherGlobalRayType *
                                                            sizeof(SceneHitgroupRecord)),
                                &global_photon_gather_record,
                                sizeof(PhotonHitgroupRecord),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(global_photon_hitgroup_record)");
          params.global_photon_handle = global_photon_gas.handle;
        }
        if (caustic_photon_count > 0) {
          build_photon_aabbs(d_caustic_photons,
                             d_caustic_photon_aabbs,
                             caustic_photon_count,
                             current_caustic_radius);
          check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(caustic_photon_aabbs)");
          caustic_photon_gas =
              build_photon_gas(context,
                               reinterpret_cast<CUdeviceptr>(d_caustic_photon_aabbs),
                               caustic_photon_count);
          caustic_photon_gather_record.data.photons = d_caustic_photons;
          caustic_photon_gather_record.data.radius = current_caustic_radius;
          check_optix(optixSbtRecordPackHeader(photon_gather_pg, &caustic_photon_gather_record),
                      "optixSbtRecordPackHeader(caustic_photon_gather_refresh)");
          check_cuda(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records +
                                                        kPhotonGatherCausticRayType *
                                                            sizeof(SceneHitgroupRecord)),
                                &caustic_photon_gather_record,
                                sizeof(PhotonHitgroupRecord),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(caustic_photon_hitgroup_record)");
          params.caustic_photon_handle = caustic_photon_gas.handle;
        }
        if (iteration == 0 && global_photon_count == 0 && caustic_photon_count == 0) {
          std::cout << "Photon GAS skipped: no stored photons\n";
        }
      }

      check_cuda(cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice),
                 "cudaMemcpy(params_render)");
      check_optix(optixLaunch(pipeline,
                              0,
                              reinterpret_cast<CUdeviceptr>(d_params),
                              sizeof(Params),
                              &render_sbt,
                              width,
                              height,
                              1),
                  "optixLaunch(render)");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(render)");

      accumulate_float4_buffer(d_output, d_accumulated_output, width * height);
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(accumulate_output)");

      if (global_photon_gas.buffer) {
        cudaFree(reinterpret_cast<void*>(global_photon_gas.buffer));
      }
      if (caustic_photon_gas.buffer) {
        cudaFree(reinterpret_cast<void*>(caustic_photon_gas.buffer));
      }

      if (use_photon_mapping && iteration + 1 < ppm_iterations) {
        const float completed = static_cast<float>(iteration + 1);
        const float shrink = std::sqrt((completed + ppm_alpha) / (completed + 1.0f));
        current_global_radius *= shrink;
        current_caustic_radius *= shrink;
      }
    }

    scale_float4_buffer(d_accumulated_output,
                        width * height,
                        1.0f / static_cast<float>(ppm_iterations));
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(scale_output)");

    const bool apply_denoiser = enable_denoiser && render_mode == kRenderModeBeauty;
    if (apply_denoiser) {
      denoise_beauty(context,
                     width,
                     height,
                     reinterpret_cast<CUdeviceptr>(d_accumulated_output),
                     reinterpret_cast<CUdeviceptr>(d_output));
    }

    std::vector<float4> accumulated_pixels(width * height);
    check_cuda(cudaMemcpy(accumulated_pixels.data(),
                          apply_denoiser ? d_output : d_accumulated_output,
                          accumulated_pixels.size() * sizeof(float4),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(accumulated_output)");
    save_ppm(output_path, accumulated_pixels, width, height);

    std::cout << "Rendered Cornell Box to " << output_path << " (" << width << "x"
              << height << ", spp=" << samples << ", photon_mapping="
              << (use_photon_mapping ? "on" : "off") << ", photons=" << last_photon_count
              << ", iterations=" << ppm_iterations
              << ", denoiser=" << (apply_denoiser ? "on" : "off") << ")\n";

    cudaFree(d_params);
    cudaFree(d_output);
    cudaFree(d_accumulated_output);
    if (d_global_photon_aabbs) {
      cudaFree(d_global_photon_aabbs);
    }
    if (d_caustic_photon_aabbs) {
      cudaFree(d_caustic_photon_aabbs);
    }
    if (d_global_photon_count) {
      cudaFree(d_global_photon_count);
    }
    if (d_caustic_photon_count) {
      cudaFree(d_caustic_photon_count);
    }
    if (d_global_photons) {
      cudaFree(d_global_photons);
    }
    if (d_caustic_photons) {
      cudaFree(d_caustic_photons);
    }
    if (d_lights) {
      cudaFree(d_lights);
    }
    cudaFree(reinterpret_cast<void*>(d_hitgroup_records));
    cudaFree(reinterpret_cast<void*>(d_miss_records));
    cudaFree(reinterpret_cast<void*>(d_raygen_records));
    cudaFree(reinterpret_cast<void*>(scene_gas.buffer));
    cudaFree(reinterpret_cast<void*>(d_materials));
    cudaFree(reinterpret_cast<void*>(d_object_indices));
    cudaFree(reinterpret_cast<void*>(d_material_indices));
    cudaFree(reinterpret_cast<void*>(d_indices));
    cudaFree(reinterpret_cast<void*>(d_normals));
    cudaFree(reinterpret_cast<void*>(d_vertices));

    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(photon_gather_pg);
    optixProgramGroupDestroy(scene_photon_pg);
    optixProgramGroupDestroy(scene_shadow_pg);
    optixProgramGroupDestroy(scene_radiance_pg);
    for (OptixProgramGroup pg : miss_pgs) {
      optixProgramGroupDestroy(pg);
    }
    optixProgramGroupDestroy(raygen_pgs[1]);
    optixProgramGroupDestroy(raygen_pgs[0]);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
