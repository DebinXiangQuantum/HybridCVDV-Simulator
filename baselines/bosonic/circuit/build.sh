#!/bin/bash

# 创建build目录
mkdir -p build
cd build

# 运行cmake
cmake ..

# 编译所有电路
make -j4

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "编译成功！"
    # 复制可执行文件到当前目录
    cp cat_state_circuit ../
    cp gkp_state_circuit ../
    cp jch_simulation_circuit ../
    cp qaoa_circuit ../
    cp qft_circuit ../
    cp shors_circuit ../
    cp state_transfer_circuit ../
    cp vqe_circuit ../
    cp run_sc26_bosonic ../
    echo "可执行文件已复制到上级目录"
else
    echo "编译失败！"
    exit 1
fi
