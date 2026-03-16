#!/bin/sh
set -eux

: "${WIDTH:=512}"
: "${HEIGHT:=512}"
: "${SPP:=1}"
: "${PHOTON_COUNT:=20000}"
: "${PHOTON_RADIUS:=35}"
: "${INDIRECT_SCALE:=1.0}"
: "${PHOTON_BOUNCES:=10}"
: "${PHOTON_NORMAL_REJECT_COS:=0.9}"
: "${GLOBAL_PHOTON_REJECTION:=0.3}"
: "${MULTI_DIFFUSE_CAUSTIC_MAP:=0}"
: "${PPM_ITERATIONS:=1}"
: "${PPM_ALPHA:=0.7}"
: "${DEBUG_STATS:=0}"
: "${RENDER_VARIANTS:=0}"
: "${PHOTON_EVAL_MODE:=0}"
: "${DIRECT_LIGHT_SAMPLES:=1}"
: "${ENABLE_DENOISER:=1}"
: "${LIGHT_RADIUS:=20}"

BUILD_DIR="build"
mkdir -p images
chmod 777 images
mkdir -p "${BUILD_DIR}"
chmod 777 "${BUILD_DIR}"
EXECUTABLE="./${BUILD_DIR}/example_app_optix"
make prepare all

# Use environment variable or default images directory
OUT_DIR="${OUTPUT_DIR:-images}"

# 1. Render Water Scene (Default)
SCENE_NAME=water "${EXECUTABLE}" 01_cornell_water.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 0 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
convert "${OUT_DIR}/01_cornell_water.ppm" "${OUT_DIR}/01_cornell_water.png"
rm -f "${OUT_DIR}/01_cornell_water.ppm"

# 2. Render Glass Scene
SCENE_NAME=glass "${EXECUTABLE}" 02_cornell_glass.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 0 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
convert "${OUT_DIR}/02_cornell_glass.ppm" "${OUT_DIR}/02_cornell_glass.png"
rm -f "${OUT_DIR}/02_cornell_glass.ppm"

if [ "${RENDER_VARIANTS}" = "1" ]; then
  SCENE_NAME=water "${EXECUTABLE}" 03_cornell_direct.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" 0 0 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
  SCENE_NAME=water "${EXECUTABLE}" 04_cornell_photon_debug.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 1 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
  SCENE_NAME=water "${EXECUTABLE}" 05_cornell_photon_indirect.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 2 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
  SCENE_NAME=water "${EXECUTABLE}" 06_cornell_caustic.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 4 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" 0 "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
  
  convert "${OUT_DIR}/03_cornell_direct.ppm" "${OUT_DIR}/03_cornell_direct.png"
  convert "${OUT_DIR}/04_cornell_photon_debug.ppm" "${OUT_DIR}/04_cornell_photon_debug.png"
  convert "${OUT_DIR}/05_cornell_photon_indirect.ppm" "${OUT_DIR}/05_cornell_photon_indirect.png"
  convert "${OUT_DIR}/06_cornell_caustic.ppm" "${OUT_DIR}/06_cornell_caustic.png"
  make clean_ppm
fi
