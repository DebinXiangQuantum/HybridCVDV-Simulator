rm -rf build
mkdir build
cd build
cmake ..
make
./HybridCVDV-Simulator_tests
./HybridCVDV-Simulator_main
./HybridCVDV-Simulator_gpu_validation