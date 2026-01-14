#!/bin/bash

# 构建脚本 - 用于构建bosonic量子操作符库和测试程序

set -e

echo "Building Bosonic Quantum Operators Library"
echo "========================================"

# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$0")

# 切换到脚本所在目录
cd "$SCRIPT_DIR"

# 创建构建目录
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

# 进入构建目录
cd "$BUILD_DIR"

# 使用CMake配置项目
echo "Configuring project with CMake..."
cmake ..

# 构建项目
echo "Building project..."
make -j$(nproc)

echo ""
echo "Build completed successfully!"
echo ""
echo "To run the test program:"
echo "  ./bosonic_test"
echo ""
echo "To install the library (optional):"
echo "  sudo make install"