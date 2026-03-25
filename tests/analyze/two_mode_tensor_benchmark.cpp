#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "cv_state_pool.h"

void initialize_vacuum_state_device(CVStatePool* pool,
                                    int state_id,
                                    int state_dim,
                                    cudaStream_t stream = nullptr,
                                    bool synchronize = true);

void apply_creation_operator_on_mode(CVStatePool* pool,
                                     const int* targets,
                                     int batch_size,
                                     int target_qumode,
                                     int num_qumodes,
                                     cudaStream_t stream = nullptr,
                                     bool synchronize = true);
void apply_beam_splitter_recursive(CVStatePool* pool,
                                   const int* targets,
                                   int batch_size,
                                   double theta,
                                   double phi,
                                   int target_qumode1,
                                   int target_qumode2,
                                   int num_qumodes,
                                   cudaStream_t stream = nullptr,
                                   bool synchronize = true);

namespace {

struct Options {
    int num_qumodes = 3;
    int cutoff = 8;
    int max_states = 512;
    int repetitions = 256;
    double theta = 0.21;
    double phi = 0.0;
};

double peak_rss_mb() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return -1.0;
    }
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

Options parse_args(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--num-qumodes") {
            options.num_qumodes = std::stoi(require_value("--num-qumodes"));
        } else if (arg == "--cutoff") {
            options.cutoff = std::stoi(require_value("--cutoff"));
        } else if (arg == "--max-states") {
            options.max_states = std::stoi(require_value("--max-states"));
        } else if (arg == "--repetitions") {
            options.repetitions = std::stoi(require_value("--repetitions"));
        } else if (arg == "--theta") {
            options.theta = std::stod(require_value("--theta"));
        } else if (arg == "--phi") {
            options.phi = std::stod(require_value("--phi"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.num_qumodes < 2) {
        throw std::invalid_argument("num-qumodes must be at least 2");
    }
    if (options.cutoff < 2) {
        throw std::invalid_argument("cutoff must be at least 2");
    }
    if (options.max_states <= 0 || options.repetitions <= 0) {
        throw std::invalid_argument("max-states and repetitions must be positive");
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        CVStatePool pool(options.cutoff, options.max_states, options.num_qumodes);

        const int state_id = pool.allocate_state();
        if (state_id < 0) {
            throw std::runtime_error("failed to allocate state");
        }
        initialize_vacuum_state_device(&pool, state_id, pool.get_max_total_dim());

        int* d_target_ids = nullptr;
        cudaError_t err = cudaMalloc(&d_target_ids, sizeof(int));
        if (err != cudaSuccess) {
            throw std::runtime_error("failed to allocate device target buffer");
        }
        err = cudaMemcpy(d_target_ids, &state_id, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_target_ids);
            throw std::runtime_error("failed to upload device target buffer");
        }

        apply_creation_operator_on_mode(&pool, d_target_ids, 1, 0, options.num_qumodes);

        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < options.repetitions; ++i) {
            const int first = i % options.num_qumodes;
            const int second = (i + 1) % options.num_qumodes;
            apply_beam_splitter_recursive(
                &pool,
                d_target_ids,
                1,
                options.theta,
                options.phi,
                first,
                second,
                options.num_qumodes);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        cudaFree(d_target_ids);

        const double wall_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        const double state_mb =
            static_cast<double>(pool.get_memory_usage()) / (1024.0 * 1024.0);

        std::cout << "=================================================\n";
        std::cout << " Two-Mode Tensor Kernel Benchmark\n";
        std::cout << "=================================================\n";
        std::cout << "qumodes=" << options.num_qumodes
                  << ", cutoff=" << options.cutoff
                  << ", repetitions=" << options.repetitions
                  << ", theta=" << options.theta
                  << ", phi=" << options.phi << '\n';
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Wall time (ms):           " << wall_ms << '\n';
        std::cout << "Avg per repetition (ms):  " << (wall_ms / options.repetitions) << '\n';
        std::cout << "State pool alloc (MB):    " << state_mb << '\n';
        std::cout << "Peak RSS (MB):            " << peak_rss_mb() << '\n';
        std::cout << "=================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "two_mode_tensor_benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
