CUDA_HOME ?= /usr/local/cuda
OPTIX_ROOT ?= $(CURDIR)/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64

SRC_DIR := src
BUILD_DIR := build
MODEL_DIR := models

CXX ?= g++
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXXFLAGS := -std=c++17 -O2 -I$(OPTIX_ROOT)/include -I$(CUDA_HOME)/include -I$(SRC_DIR)
LDFLAGS := -L$(CUDA_HOME)/lib64 -lcudart -lcuda -ldl
NVCCFLAGS := -std=c++17 -O2 -I$(OPTIX_ROOT)/include -I$(CUDA_HOME)/include -I$(SRC_DIR)

TARGET := $(BUILD_DIR)/example_app_optix
PTX := $(BUILD_DIR)/renderer.ptx
CUDA_OBJ := $(BUILD_DIR)/photon_helpers.o
MESH001 := $(MODEL_DIR)/Mesh001.ply
MESH00001 := $(MODEL_DIR)/mesh_00001.ply
MESH001_URL := https://raw.githubusercontent.com/SirKero/RTProgressivePhotonMapper/master/Scenes/water-caustic/models/Mesh001.ply
MESH00001_URL := https://raw.githubusercontent.com/SirKero/RTProgressivePhotonMapper/master/Scenes/caustic-glass/geometry/mesh_00001.ply

STB_INC := -I$(OPTIX_ROOT)/SDK/support/tinygltf

all: $(TARGET)

prepare: $(MESH001) $(MESH00001)

$(TARGET): $(SRC_DIR)/main.cpp $(SRC_DIR)/scene.cpp $(PTX) $(CUDA_OBJ) $(SRC_DIR)/shared_defs.h $(SRC_DIR)/photon_helpers.h $(SRC_DIR)/scene.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/main.cpp $(SRC_DIR)/scene.cpp $(CUDA_OBJ) $(LDFLAGS)

$(PTX): $(SRC_DIR)/renderer.cu $(SRC_DIR)/shared_defs.h $(SRC_DIR)/raytracer.cu $(SRC_DIR)/scene_ray_gen.cu $(SRC_DIR)/raytracer_helpers.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --ptx -o $@ $<

$(CUDA_OBJ): $(SRC_DIR)/photon_helpers.cu $(SRC_DIR)/photon_helpers.h $(SRC_DIR)/shared_defs.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(MODEL_DIR):
	mkdir -p $(MODEL_DIR)

$(MESH001): | $(MODEL_DIR)
	curl -L --fail "$(MESH001_URL)" -o $@

$(MESH00001): | $(MODEL_DIR)
	curl -L --fail "$(MESH00001_URL)" -o $@

clean:
	rm -rf $(BUILD_DIR)

clean_all: clean clean_png clean_ppm

clean_png:
	rm -rf images/*.png

clean_ppm:
	rm -rf images/*.ppm

.PHONY: all prepare clean clean_all
