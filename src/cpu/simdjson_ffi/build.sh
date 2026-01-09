#!/bin/bash
# Build script for simdjson FFI wrapper
# Uses simdjson installed via pixi (conda-forge)
# This script is idempotent - skips rebuild if library is up-to-date
#
# NOTE: When used as activation script, avoid 'exit' which breaks shell setup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../../../build"
TARGET="$BUILD_DIR/libsimdjson_wrapper.so"
SOURCE="$SCRIPT_DIR/simdjson_wrapper.cpp"
HEADER="$SCRIPT_DIR/simdjson_wrapper.h"

# Verify CONDA_PREFIX is set (pixi sets this)
if [ -z "$CONDA_PREFIX" ]; then
    echo "Warning: CONDA_PREFIX not set. Skipping FFI build."
    return 0 2>/dev/null || true
fi

# Check if rebuild is needed
_needs_rebuild() {
    # Rebuild if target doesn't exist
    [ ! -f "$TARGET" ] && return 0

    # Rebuild if source files are newer than target
    [ "$SOURCE" -nt "$TARGET" ] && return 0
    [ "$HEADER" -nt "$TARGET" ] && return 0

    # Rebuild if simdjson library is newer (in case of pixi update)
    [ "$CONDA_PREFIX/lib/libsimdjson.so" -nt "$TARGET" ] && return 0

    return 1
}

if ! _needs_rebuild; then
    # Already up-to-date, skip silently
    return 0 2>/dev/null || true
fi

echo "========================================"
echo "Building simdjson FFI wrapper"
echo "========================================"
echo ""
echo "Using simdjson from: $CONDA_PREFIX"
echo "  Header: $CONDA_PREFIX/include/simdjson.h"
echo "  Library: $CONDA_PREFIX/lib/libsimdjson.so"
echo ""

# Verify simdjson is installed
if [ ! -f "$CONDA_PREFIX/include/simdjson.h" ]; then
    echo "Error: simdjson.h not found at $CONDA_PREFIX/include/"
    echo "Run 'pixi install' to install dependencies"
    return 1 2>/dev/null || true
fi

# Create build directory
mkdir -p "$BUILD_DIR"

echo "Building libsimdjson_wrapper.so..."

# Build the wrapper library, linking against installed simdjson
if g++ -O3 -std=c++17 -fPIC -DNDEBUG -shared \
    -o "$TARGET" \
    "$SOURCE" \
    -I"$CONDA_PREFIX/include" \
    -L"$CONDA_PREFIX/lib" \
    -lsimdjson \
    -Wl,-rpath,"$CONDA_PREFIX/lib"; then
    echo ""
    echo "Build complete!"
    echo "Library: $TARGET"
    ls -la "$TARGET"
else
    echo "Build failed!"
    return 1 2>/dev/null || true
fi
