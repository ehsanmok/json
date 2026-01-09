# Makefile for building simdjson FFI shared library
# Uses simdjson installed via pixi (conda-forge)
# Recommended: use 'pixi run build' instead

CXX := g++
CXXFLAGS := -O3 -std=c++17 -fPIC -DNDEBUG

SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(SCRIPT_DIR)../../../build

TARGET := $(BUILD_DIR)/libsimdjson_wrapper.so

# Use CONDA_PREFIX from pixi environment
ifndef CONDA_PREFIX
$(error CONDA_PREFIX not set. Run via 'pixi run build' or activate pixi environment)
endif

INCLUDES := -I$(CONDA_PREFIX)/include
LIBS := -L$(CONDA_PREFIX)/lib -lsimdjson -Wl,-rpath,$(CONDA_PREFIX)/lib

.PHONY: all clean check-simdjson

all: check-simdjson $(TARGET)

check-simdjson:
	@if [ ! -f "$(CONDA_PREFIX)/include/simdjson.h" ]; then \
		echo "Error: simdjson.h not found at $(CONDA_PREFIX)/include/"; \
		echo "Run 'pixi install' to install dependencies"; \
		exit 1; \
	fi

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): simdjson_wrapper.cpp simdjson_wrapper.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -o $@ simdjson_wrapper.cpp $(INCLUDES) $(LIBS)

clean:
	rm -f $(TARGET)
