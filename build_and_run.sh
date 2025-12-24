rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug .. -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc
make
gdb -batch -ex "run" -ex "bt" ./HybridCVDV-Simulator_tests
./HybridCVDV-Simulator_main