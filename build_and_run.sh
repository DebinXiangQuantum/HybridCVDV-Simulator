rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb -batch -ex "run" -ex "bt" ./HybridCVDV-Simulator_tests
./HybridCVDV-Simulator_main
./HybridCVDV-Simulator_gpu_validation