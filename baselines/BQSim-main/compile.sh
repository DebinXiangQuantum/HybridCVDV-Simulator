#!/bin/bash

# 跳过 qpp 安装，直接编译 BQSim
cd /root/Hybirdcvdv/baselines/BQSim-main
mkdir -p build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j9
