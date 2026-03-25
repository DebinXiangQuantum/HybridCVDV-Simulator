#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build-codex}"
OUT_DIR="${2:-$ROOT_DIR/bench-results/nsight-suite}"
JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)}"

mkdir -p "$OUT_DIR"

cmake --build "$BUILD_DIR" \
  --target \
    HybridCVDV-Simulator_random_circuit_benchmark \
    HybridCVDV-Simulator_two_mode_tensor_benchmark \
    HybridCVDV-Simulator_cuda_graph_benchmark \
  -j "$JOBS"

RANDOM_EXE="$BUILD_DIR/HybridCVDV-Simulator_random_circuit_benchmark"
TWO_MODE_EXE="$BUILD_DIR/HybridCVDV-Simulator_two_mode_tensor_benchmark"
GRAPH_EXE="$BUILD_DIR/HybridCVDV-Simulator_cuda_graph_benchmark"

RANDOM_BASE="$OUT_DIR/random_circuit_mixed"
GRAPH_BASE="$OUT_DIR/cuda_graph"
MEMORY_CSV="$OUT_DIR/two_mode_memory_workload.csv"
RANDOM_MEMORY_CSV="$OUT_DIR/random_circuit_memory_workload.csv"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --output "$RANDOM_BASE" \
  "$RANDOM_EXE" \
  --seed 20260315 \
  --cutoff 8 \
  --total-gates 72

nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output "$RANDOM_BASE" \
  "$RANDOM_BASE.nsys-rep"

ncu \
  --target-processes all \
  --launch-skip 2 \
  --launch-count 1 \
  --section MemoryWorkloadAnalysis \
  --csv \
  "$TWO_MODE_EXE" \
  --num-qumodes 3 \
  --cutoff 8 \
  --repetitions 256 \
  > "$MEMORY_CSV"

ncu \
  --target-processes all \
  --launch-skip 4 \
  --launch-count 1 \
  --section MemoryWorkloadAnalysis \
  --csv \
  "$RANDOM_EXE" \
  --seed 20260315 \
  --cutoff 8 \
  --total-gates 72 \
  > "$RANDOM_MEMORY_CSV"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --output "$GRAPH_BASE" \
  "$GRAPH_EXE" \
  --num-states 256 \
  --replays 25

nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output "$GRAPH_BASE" \
  "$GRAPH_BASE.nsys-rep"

# Memory Workload Analysis for CUDA Graph benchmark
GRAPH_MEMORY_CSV="$OUT_DIR/cuda_graph_memory_workload.csv"
ncu \
  --target-processes all \
  --launch-skip 0 \
  --launch-count 1 \
  --section MemoryWorkloadAnalysis \
  --replay-mode application \
  --csv \
  "$GRAPH_EXE" \
  --num-states 256 \
  --replays 5 \
  > "$GRAPH_MEMORY_CSV"

# Compute workload (roofline) for random circuit
RANDOM_COMPUTE_CSV="$OUT_DIR/random_circuit_compute_workload.csv"
ncu \
  --target-processes all \
  --launch-skip 4 \
  --launch-count 1 \
  --section ComputeWorkloadAnalysis \
  --section SpeedOfLight \
  --csv \
  "$RANDOM_EXE" \
  --seed 20260315 \
  --cutoff 8 \
  --total-gates 72 \
  > "$RANDOM_COMPUTE_CSV"

if command -v sqlite3 >/dev/null 2>&1; then
  RANDOM_GRAPH_CALLS="$(sqlite3 "$RANDOM_BASE.sqlite" "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL;" 2>/dev/null || echo "N/A")"
  CUDA_GRAPH_CALLS="$(sqlite3 "$GRAPH_BASE.sqlite" "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL;" 2>/dev/null || echo "N/A")"
  echo "random benchmark graph-backed kernel launches: $RANDOM_GRAPH_CALLS"
  echo "cuda-graph benchmark graph-backed kernel launches: $CUDA_GRAPH_CALLS"
fi

echo ""
echo "Nsight artifacts written to: $OUT_DIR"
echo "  Timeline (random):        $RANDOM_BASE.nsys-rep"
echo "  Timeline (cuda graph):    $GRAPH_BASE.nsys-rep"
echo "  Memory workload (random): $RANDOM_MEMORY_CSV"
echo "  Memory workload (graph):  $GRAPH_MEMORY_CSV"
echo "  Compute workload (random):$RANDOM_COMPUTE_CSV"
echo "  Two-mode memory:          $MEMORY_CSV"
