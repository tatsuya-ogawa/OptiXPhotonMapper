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
"${EXECUTABLE}" cornell_photon.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 0 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"

# Use environment variable or default images directory
OUT_DIR="${OUTPUT_DIR:-images}"
convert "${OUT_DIR}/cornell_photon.ppm" "${OUT_DIR}/cornell_photon.png"
rm -f "${OUT_DIR}/cornell_photon.ppm"

if [ "${RENDER_VARIANTS}" = "1" ]; then
  "${EXECUTABLE}" cornell_direct.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" 0 0 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
  "${EXECUTABLE}" cornell_photon_debug.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 1 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
  "${EXECUTABLE}" cornell_photon_indirect.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 2 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" "${ENABLE_DENOISER}" "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
  "${EXECUTABLE}" cornell_caustic.ppm "${WIDTH}" "${HEIGHT}" "${SPP}" "${PHOTON_COUNT}" 4 "${PHOTON_RADIUS}" "${INDIRECT_SCALE}" "${PHOTON_BOUNCES}" "${PHOTON_NORMAL_REJECT_COS}" "${PPM_ITERATIONS}" "${PPM_ALPHA}" "${DEBUG_STATS}" "${PHOTON_EVAL_MODE}" "${DIRECT_LIGHT_SAMPLES}" 0 "${LIGHT_RADIUS}" "${GLOBAL_PHOTON_REJECTION}" "${MULTI_DIFFUSE_CAUSTIC_MAP}"
  
  convert "${OUT_DIR}/cornell_direct.ppm" "${OUT_DIR}/cornell_direct.png"
  convert "${OUT_DIR}/cornell_photon_debug.ppm" "${OUT_DIR}/cornell_photon_debug.png"
  convert "${OUT_DIR}/cornell_photon_indirect.ppm" "${OUT_DIR}/cornell_photon_indirect.png"
  convert "${OUT_DIR}/cornell_caustic.ppm" "${OUT_DIR}/cornell_caustic.png"
  make clean_ppm
fi
