#!/bin/bash

# 创建build目录
mkdir -p build
cd build

# 运行cmake
cmake ..

# 编译所有电路
make -j4

