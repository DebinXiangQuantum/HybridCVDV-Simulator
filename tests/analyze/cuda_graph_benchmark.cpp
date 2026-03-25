#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "batch_scheduler.h"
#include "cv_state_pool.h"

void initialize_vacuum_state_device(CVStatePool* pool, int state_id, int state_dim);

namespace {

struct Options {
    int num_qumodes = 2;
    int cutoff = 8;
    int num_states = 256;
    int replays = 25;
    double theta = 0.125;
    int target_qumode = 0;
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
        } else if (arg == "--num-states") {
            options.num_states = std::stoi(require_value("--num-states"));
        } else if (arg == "--replays") {
            options.replays = std::stoi(require_value("--replays"));
        } else if (arg == "--theta") {
            options.theta = std::stod(require_value("--theta"));
        } else if (arg == "--target-qumode") {
            options.target_qumode = std::stoi(require_value("--target-qumode"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.num_qumodes < 1) {
        throw std::invalid_argument("num-qumodes must be positive");
    }
    if (options.cutoff < 2) {
        throw std::invalid_argument("cutoff must be at least 2");
    }
    if (options.num_states < 1 || options.replays < 1) {
        throw std::invalid_argument("num-states and replays must be positive");
    }
    if (options.target_qumode < 0 || options.target_qumode >= options.num_qumodes) {
        throw std::invalid_argument("target-qumode out of range");
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        CVStatePool pool(options.cutoff, options.num_states + 8, options.num_qumodes);
        const int64_t state_dim = pool.get_max_total_dim();

        std::vector<int> state_ids;
        state_ids.reserve(options.num_states);
        for (int i = 0; i < options.num_states; ++i) {
            const int state_id = pool.allocate_state();
            if (state_id < 0) {
                throw std::runtime_error("failed to allocate benchmark state");
            }
            initialize_vacuum_state_device(&pool, state_id, state_dim);
            state_ids.push_back(state_id);
        }

        BatchScheduler scheduler(&pool, 1024);
        scheduler.add_task(
            BatchTask(
                GateType::PHASE_ROTATION,
                state_ids,
                {std::complex<double>(options.theta, 0.0)},
                0,
                {},
                {options.target_qumode}));

        const auto capture_start = std::chrono::high_resolution_clock::now();
        scheduler.execute_with_cuda_graph();
        const auto capture_end = std::chrono::high_resolution_clock::now();

        const auto replay_start = std::chrono::high_resolution_clock::now();
        for (int replay = 1; replay < options.replays; ++replay) {
            scheduler.execute_with_cuda_graph();
        }
        const auto replay_end = std::chrono::high_resolution_clock::now();

        const double capture_ms =
            std::chrono::duration<double, std::milli>(capture_end - capture_start).count();
        const double replay_ms =
            std::chrono::duration<double, std::milli>(replay_end - replay_start).count();
        const double avg_replay_ms =
            options.replays > 1 ? replay_ms / static_cast<double>(options.replays - 1) : 0.0;

        std::cout << "=================================================\n";
        std::cout << " CUDA Graph Benchmark\n";
        std::cout << "=================================================\n";
        std::cout << "qumodes=" << options.num_qumodes
                  << ", cutoff=" << options.cutoff
                  << ", num_states=" << options.num_states
                  << ", replays=" << options.replays
                  << ", theta=" << options.theta
                  << ", target_qumode=" << options.target_qumode << '\n';
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Capture+first launch (ms): " << capture_ms << '\n';
        std::cout << "Replay total (ms):         " << replay_ms << '\n';
        std::cout << "Replay avg (ms):           " << avg_replay_ms << '\n';
        std::cout << "State pool alloc (MB):     "
                  << static_cast<double>(pool.get_memory_usage()) / (1024.0 * 1024.0) << '\n';
        std::cout << "Peak RSS (MB):             " << peak_rss_mb() << '\n';
        std::cout << "=================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cuda_graph_benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
