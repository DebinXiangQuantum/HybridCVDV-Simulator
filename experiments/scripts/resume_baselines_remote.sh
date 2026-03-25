#!/bin/bash
set -e
cd /home/xiangdebin/sandboxes/HybridCVDV-Simulator
mkdir -p experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316
mkdir -p checkpoints

CHECKPOINT="checkpoints/sc26_baselines_hybrid_20260316.json"
if [ ! -f "$CHECKPOINT" ]; then
    echo "{\"completed\": []}" > "$CHECKPOINT"
fi


# Case: strawberryfields_tf_vqe_circuit_nq3_nm3_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq3_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq3_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq3_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq3_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq3_nm3_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq3_nm3_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq3_nm3_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq3_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq3_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq3_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq3_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq3_nm3_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq3_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq3_nm3_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq3_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq3_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq3_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq3_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq3_nm3_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq3_nm3_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq3_nm3_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq3_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq3_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq3_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq3_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq3_nm3_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq3_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq3_nm4_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq3_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq3_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq3_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq3_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq3_nm4_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq3_nm4_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq3_nm4_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq3_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq3_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq3_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq3_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq3_nm4_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq3_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq3_nm4_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq3_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq3_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq3_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq3_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq3_nm4_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq3_nm4_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq3_nm4_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq3_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq3_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq3_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq3_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq3_nm4_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq3_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq3_nm5_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq3_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq3_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq3_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq3_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq3_nm5_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq3_nm5_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq3_nm5_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq3_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq3_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq3_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq3_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq3_nm5_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq3_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq3_nm5_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq3_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq3_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq3_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq3_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq3_nm5_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq3_nm5_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq3_nm5_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq3_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq3_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq3_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq3_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq3_nm5_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq3_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq3_nm6_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq3_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq3_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq3_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq3_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq3_nm6_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq3_nm6_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq3_nm6_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq3_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq3_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq3_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq3_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq3_nm6_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq3_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq3_nm6_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq3_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq3_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq3_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq3_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq3_nm6_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq3_nm6_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq3_nm6_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq3_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq3_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq3_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq3_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq3_nm6_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq3_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq3_nm7_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq3_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq3_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq3_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq3_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq3_nm7_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq3_nm7_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq3_nm7_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq3_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq3_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq3_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq3_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq3_nm7_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq3_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq3_nm7_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq3_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq3_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq3_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq3_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq3_nm7_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq3_nm7_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq3_nm7_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq3_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq3_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 3 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq3_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq3_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq3_nm7_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq3_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq4_nm3_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq4_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq4_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq4_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq4_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq4_nm3_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq4_nm3_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq4_nm3_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq4_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq4_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq4_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq4_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq4_nm3_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq4_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq4_nm3_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq4_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq4_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq4_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq4_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq4_nm3_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq4_nm3_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq4_nm3_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq4_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq4_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq4_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq4_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq4_nm3_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq4_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq4_nm4_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq4_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq4_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq4_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq4_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq4_nm4_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq4_nm4_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq4_nm4_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq4_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq4_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq4_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq4_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq4_nm4_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq4_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq4_nm4_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq4_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq4_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq4_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq4_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq4_nm4_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq4_nm4_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq4_nm4_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq4_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq4_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq4_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq4_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq4_nm4_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq4_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq4_nm5_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq4_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq4_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq4_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq4_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq4_nm5_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq4_nm5_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq4_nm5_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq4_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq4_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq4_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq4_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq4_nm5_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq4_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq4_nm5_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq4_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq4_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq4_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq4_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq4_nm5_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq4_nm5_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq4_nm5_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq4_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq4_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq4_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq4_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq4_nm5_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq4_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq4_nm6_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq4_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq4_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq4_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq4_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq4_nm6_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq4_nm6_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq4_nm6_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq4_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq4_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq4_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq4_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq4_nm6_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq4_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq4_nm6_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq4_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq4_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq4_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq4_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq4_nm6_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq4_nm6_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq4_nm6_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq4_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq4_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq4_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq4_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq4_nm6_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq4_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq4_nm7_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq4_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq4_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq4_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq4_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq4_nm7_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq4_nm7_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq4_nm7_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq4_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq4_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq4_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq4_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq4_nm7_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq4_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq4_nm7_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq4_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq4_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq4_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq4_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq4_nm7_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq4_nm7_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq4_nm7_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq4_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq4_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq4_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq4_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq4_nm7_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq4_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq5_nm3_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq5_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq5_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq5_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq5_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq5_nm3_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq5_nm3_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq5_nm3_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq5_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq5_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq5_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq5_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq5_nm3_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq5_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq5_nm3_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq5_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq5_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq5_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq5_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq5_nm3_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq5_nm3_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq5_nm3_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq5_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq5_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq5_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq5_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq5_nm3_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq5_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq5_nm4_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq5_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq5_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq5_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq5_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq5_nm4_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq5_nm4_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq5_nm4_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq5_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq5_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq5_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq5_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq5_nm4_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq5_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq5_nm4_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq5_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq5_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq5_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq5_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq5_nm4_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq5_nm4_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq5_nm4_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq5_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq5_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq5_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq5_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq5_nm4_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq5_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq5_nm5_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq5_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq5_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq5_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq5_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq5_nm5_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq5_nm5_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq5_nm5_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq5_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq5_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq5_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq5_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq5_nm5_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq5_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq5_nm5_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq5_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq5_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq5_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq5_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq5_nm5_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq5_nm5_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq5_nm5_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq5_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq5_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq5_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq5_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq5_nm5_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq5_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq5_nm6_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq5_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq5_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq5_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq5_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq5_nm6_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq5_nm6_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq5_nm6_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq5_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq5_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq5_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq5_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq5_nm6_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq5_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq5_nm6_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq5_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq5_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq5_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq5_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq5_nm6_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq5_nm6_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq5_nm6_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq5_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq5_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq5_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq5_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq5_nm6_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq5_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq5_nm7_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq5_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq5_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq5_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq5_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq5_nm7_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq5_nm7_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq5_nm7_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq5_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq5_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq5_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq5_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq5_nm7_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq5_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq5_nm7_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq5_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq5_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq5_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq5_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq5_nm7_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq5_nm7_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq5_nm7_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq5_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq5_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq5_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq5_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq5_nm7_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq5_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq6_nm3_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq6_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq6_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq6_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq6_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq6_nm3_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq6_nm3_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq6_nm3_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq6_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq6_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq6_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq6_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq6_nm3_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq6_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq6_nm3_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq6_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq6_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq6_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq6_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq6_nm3_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq6_nm3_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq6_nm3_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq6_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq6_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq6_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq6_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq6_nm3_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq6_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq6_nm4_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq6_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq6_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq6_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq6_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq6_nm4_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq6_nm4_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq6_nm4_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq6_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq6_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq6_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq6_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq6_nm4_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq6_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq6_nm4_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq6_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq6_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq6_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq6_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq6_nm4_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq6_nm4_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq6_nm4_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq6_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq6_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq6_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq6_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq6_nm4_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq6_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq6_nm5_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq6_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq6_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq6_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq6_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq6_nm5_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq6_nm5_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq6_nm5_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq6_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq6_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq6_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq6_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq6_nm5_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq6_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq6_nm5_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq6_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq6_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq6_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq6_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq6_nm5_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq6_nm5_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq6_nm5_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq6_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq6_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq6_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq6_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq6_nm5_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq6_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq6_nm6_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq6_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq6_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq6_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq6_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq6_nm6_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq6_nm6_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq6_nm6_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq6_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq6_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq6_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq6_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq6_nm6_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq6_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq6_nm6_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq6_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq6_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq6_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq6_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq6_nm6_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq6_nm6_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq6_nm6_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq6_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq6_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq6_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq6_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq6_nm6_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq6_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq6_nm7_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq6_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq6_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq6_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq6_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq6_nm7_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq6_nm7_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq6_nm7_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq6_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq6_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq6_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq6_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq6_nm7_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq6_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq6_nm7_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq6_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq6_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq6_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq6_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq6_nm7_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq6_nm7_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq6_nm7_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq6_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq6_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 6 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq6_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq6_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq6_nm7_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq6_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq7_nm3_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq7_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq7_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq7_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq7_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq7_nm3_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq7_nm3_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq7_nm3_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq7_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq7_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq7_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq7_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq7_nm3_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq7_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq7_nm3_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq7_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq7_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq7_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq7_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq7_nm3_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq7_nm3_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq7_nm3_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq7_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq7_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq7_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq7_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq7_nm3_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq7_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq7_nm4_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq7_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq7_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq7_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq7_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq7_nm4_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq7_nm4_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq7_nm4_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq7_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq7_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq7_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq7_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq7_nm4_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq7_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq7_nm4_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq7_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq7_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq7_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq7_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq7_nm4_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq7_nm4_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq7_nm4_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq7_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq7_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq7_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq7_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq7_nm4_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq7_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq7_nm5_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq7_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq7_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq7_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq7_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq7_nm5_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq7_nm5_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq7_nm5_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq7_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq7_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq7_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq7_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq7_nm5_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq7_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq7_nm5_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq7_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq7_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq7_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq7_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq7_nm5_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq7_nm5_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq7_nm5_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq7_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq7_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq7_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq7_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq7_nm5_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq7_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq7_nm6_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq7_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq7_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq7_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq7_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq7_nm6_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq7_nm6_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq7_nm6_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq7_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq7_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq7_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq7_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq7_nm6_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq7_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq7_nm6_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq7_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq7_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq7_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq7_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq7_nm6_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq7_nm6_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq7_nm6_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq7_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq7_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq7_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq7_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq7_nm6_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq7_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq7_nm7_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq7_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq7_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq7_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq7_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq7_nm7_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq7_nm7_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq7_nm7_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq7_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq7_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq7_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq7_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq7_nm7_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq7_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq7_nm7_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq7_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq7_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq7_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq7_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq7_nm7_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq7_nm7_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq7_nm7_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq7_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq7_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 7 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq7_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq7_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq7_nm7_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq7_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq8_nm3_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq8_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq8_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq8_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq8_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq8_nm3_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq8_nm3_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq8_nm3_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq8_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq8_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq8_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq8_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq8_nm3_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq8_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq8_nm3_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq8_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq8_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq8_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq8_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq8_nm3_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq8_nm3_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq8_nm3_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq8_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq8_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq8_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq8_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq8_nm3_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq8_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq8_nm4_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq8_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq8_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq8_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq8_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq8_nm4_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq8_nm4_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq8_nm4_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq8_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq8_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq8_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq8_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq8_nm4_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq8_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq8_nm4_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq8_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq8_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq8_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq8_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq8_nm4_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq8_nm4_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq8_nm4_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq8_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq8_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq8_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq8_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq8_nm4_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq8_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq8_nm5_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq8_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq8_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq8_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq8_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq8_nm5_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq8_nm5_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq8_nm5_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq8_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq8_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq8_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq8_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq8_nm5_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq8_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq8_nm5_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq8_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq8_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq8_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq8_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq8_nm5_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq8_nm5_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq8_nm5_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq8_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq8_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq8_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq8_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq8_nm5_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq8_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq8_nm6_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq8_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq8_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq8_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq8_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq8_nm6_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq8_nm6_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq8_nm6_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq8_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq8_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq8_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq8_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq8_nm6_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq8_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq8_nm6_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq8_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq8_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq8_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq8_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq8_nm6_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq8_nm6_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq8_nm6_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq8_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq8_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq8_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq8_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq8_nm6_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq8_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq8_nm7_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq8_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq8_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq8_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq8_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq8_nm7_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq8_nm7_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq8_nm7_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq8_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq8_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq8_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq8_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq8_nm7_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq8_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq8_nm7_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq8_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq8_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq8_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq8_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq8_nm7_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq8_nm7_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq8_nm7_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq8_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq8_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq8_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq8_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq8_nm7_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq8_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq9_nm3_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq9_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq9_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq9_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq9_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq9_nm3_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq9_nm3_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq9_nm3_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq9_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq9_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq9_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq9_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq9_nm3_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq9_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq9_nm3_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq9_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq9_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq9_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq9_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq9_nm3_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq9_nm3_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq9_nm3_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq9_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq9_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq9_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq9_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq9_nm3_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq9_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq9_nm4_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq9_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq9_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq9_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq9_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq9_nm4_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq9_nm4_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq9_nm4_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq9_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq9_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq9_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq9_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq9_nm4_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq9_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq9_nm4_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq9_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq9_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq9_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq9_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq9_nm4_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq9_nm4_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq9_nm4_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq9_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq9_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq9_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq9_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq9_nm4_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq9_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq9_nm5_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq9_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq9_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq9_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq9_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq9_nm5_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq9_nm5_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq9_nm5_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq9_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq9_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq9_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq9_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq9_nm5_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq9_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq9_nm5_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq9_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq9_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq9_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq9_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq9_nm5_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq9_nm5_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq9_nm5_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq9_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq9_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq9_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq9_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq9_nm5_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq9_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq9_nm6_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq9_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq9_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq9_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq9_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq9_nm6_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq9_nm6_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq9_nm6_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq9_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq9_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq9_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq9_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq9_nm6_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq9_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq9_nm6_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq9_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq9_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq9_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq9_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq9_nm6_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq9_nm6_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq9_nm6_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq9_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq9_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq9_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq9_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq9_nm6_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq9_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq9_nm7_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq9_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq9_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq9_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq9_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq9_nm7_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq9_nm7_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq9_nm7_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq9_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq9_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq9_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq9_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq9_nm7_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq9_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq9_nm7_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq9_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq9_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq9_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq9_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq9_nm7_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq9_nm7_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq9_nm7_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq9_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq9_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 9 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq9_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq9_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq9_nm7_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq9_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq10_nm3_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq10_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq10_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq10_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq10_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq10_nm3_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq10_nm3_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq10_nm3_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq10_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq10_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 3 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq10_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq10_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq10_nm3_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq10_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq10_nm3_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq10_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq10_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq10_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq10_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq10_nm3_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq10_nm3_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq10_nm3_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq10_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq10_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 3 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq10_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq10_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq10_nm3_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq10_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq10_nm4_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq10_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq10_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq10_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq10_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq10_nm4_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq10_nm4_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq10_nm4_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq10_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq10_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 4 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq10_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq10_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq10_nm4_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq10_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq10_nm4_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq10_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq10_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq10_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq10_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq10_nm4_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq10_nm4_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq10_nm4_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq10_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq10_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 4 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq10_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq10_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq10_nm4_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq10_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq10_nm5_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq10_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq10_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq10_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq10_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq10_nm5_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq10_nm5_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq10_nm5_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq10_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq10_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 5 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq10_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq10_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq10_nm5_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq10_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq10_nm5_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq10_nm5_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq10_nm5_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq10_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq10_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq10_nm5_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq10_nm5_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq10_nm5_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq10_nm5_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq10_nm5_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 5 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq10_nm5_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq10_nm5_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq10_nm5_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq10_nm5_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq10_nm6_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq10_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq10_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq10_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq10_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq10_nm6_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq10_nm6_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq10_nm6_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq10_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq10_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 6 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq10_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq10_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq10_nm6_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq10_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq10_nm6_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq10_nm6_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq10_nm6_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq10_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq10_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq10_nm6_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq10_nm6_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq10_nm6_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq10_nm6_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq10_nm6_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 6 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq10_nm6_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq10_nm6_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq10_nm6_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq10_nm6_c16, already completed."
fi

# Case: strawberryfields_tf_vqe_circuit_nq10_nm7_c16
if ! grep -q '"strawberryfields_tf_vqe_circuit_nq10_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_vqe_circuit_nq10_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_vqe_circuit_nq10_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_vqe_circuit_nq10_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_vqe_circuit_nq10_nm7_c16"
else
    echo "Skipping strawberryfields_tf_vqe_circuit_nq10_nm7_c16, already completed."
fi

# Case: mrmustard_jax_vqe_circuit_nq10_nm7_c16
if ! grep -q '"mrmustard_jax_vqe_circuit_nq10_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_vqe_circuit_nq10_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload vqe_circuit --cutoff 16 --num-modes 7 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --layers 2 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_vqe_circuit_nq10_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_vqe_circuit_nq10_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_vqe_circuit_nq10_nm7_c16"
else
    echo "Skipping mrmustard_jax_vqe_circuit_nq10_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_jch_simulation_circuit_nq10_nm7_c16
if ! grep -q '"strawberryfields_tf_jch_simulation_circuit_nq10_nm7_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_jch_simulation_circuit_nq10_nm7_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_jch_simulation_circuit_nq10_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_jch_simulation_circuit_nq10_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_jch_simulation_circuit_nq10_nm7_c16"
else
    echo "Skipping strawberryfields_tf_jch_simulation_circuit_nq10_nm7_c16, already completed."
fi

# Case: mrmustard_jax_jch_simulation_circuit_nq10_nm7_c16
if ! grep -q '"mrmustard_jax_jch_simulation_circuit_nq10_nm7_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_jch_simulation_circuit_nq10_nm7_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload jch_simulation_circuit --cutoff 16 --num-modes 7 --num-qubits 10 --warmup-runs 2 --measured-runs 5 --timesteps 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_jch_simulation_circuit_nq10_nm7_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_jch_simulation_circuit_nq10_nm7_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_jch_simulation_circuit_nq10_nm7_c16"
else
    echo "Skipping mrmustard_jax_jch_simulation_circuit_nq10_nm7_c16, already completed."
fi

# Case: strawberryfields_tf_cat_state_circuit_nq1_nm1_c16
if ! grep -q '"strawberryfields_tf_cat_state_circuit_nq1_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_cat_state_circuit_nq1_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload cat_state_circuit --cutoff 16 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_cat_state_circuit_nq1_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_cat_state_circuit_nq1_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_cat_state_circuit_nq1_nm1_c16"
else
    echo "Skipping strawberryfields_tf_cat_state_circuit_nq1_nm1_c16, already completed."
fi

# Case: mrmustard_jax_cat_state_circuit_nq1_nm1_c16
if ! grep -q '"mrmustard_jax_cat_state_circuit_nq1_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_cat_state_circuit_nq1_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload cat_state_circuit --cutoff 16 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_cat_state_circuit_nq1_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_cat_state_circuit_nq1_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_cat_state_circuit_nq1_nm1_c16"
else
    echo "Skipping mrmustard_jax_cat_state_circuit_nq1_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_cat_state_circuit_nq1_nm1_c32
if ! grep -q '"strawberryfields_tf_cat_state_circuit_nq1_nm1_c32"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_cat_state_circuit_nq1_nm1_c32..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload cat_state_circuit --cutoff 32 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_cat_state_circuit_nq1_nm1_c32.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_cat_state_circuit_nq1_nm1_c32"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_cat_state_circuit_nq1_nm1_c32"
else
    echo "Skipping strawberryfields_tf_cat_state_circuit_nq1_nm1_c32, already completed."
fi

# Case: mrmustard_jax_cat_state_circuit_nq1_nm1_c32
if ! grep -q '"mrmustard_jax_cat_state_circuit_nq1_nm1_c32"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_cat_state_circuit_nq1_nm1_c32..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload cat_state_circuit --cutoff 32 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_cat_state_circuit_nq1_nm1_c32.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_cat_state_circuit_nq1_nm1_c32"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_cat_state_circuit_nq1_nm1_c32"
else
    echo "Skipping mrmustard_jax_cat_state_circuit_nq1_nm1_c32, already completed."
fi

# Case: strawberryfields_tf_gkp_state_circuit_nq1_nm1_c16
if ! grep -q '"strawberryfields_tf_gkp_state_circuit_nq1_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_gkp_state_circuit_nq1_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload gkp_state_circuit --cutoff 16 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_gkp_state_circuit_nq1_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_gkp_state_circuit_nq1_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_gkp_state_circuit_nq1_nm1_c16"
else
    echo "Skipping strawberryfields_tf_gkp_state_circuit_nq1_nm1_c16, already completed."
fi

# Case: mrmustard_jax_gkp_state_circuit_nq1_nm1_c16
if ! grep -q '"mrmustard_jax_gkp_state_circuit_nq1_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_gkp_state_circuit_nq1_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload gkp_state_circuit --cutoff 16 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_gkp_state_circuit_nq1_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_gkp_state_circuit_nq1_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_gkp_state_circuit_nq1_nm1_c16"
else
    echo "Skipping mrmustard_jax_gkp_state_circuit_nq1_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_gkp_state_circuit_nq1_nm1_c32
if ! grep -q '"strawberryfields_tf_gkp_state_circuit_nq1_nm1_c32"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_gkp_state_circuit_nq1_nm1_c32..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload gkp_state_circuit --cutoff 32 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_gkp_state_circuit_nq1_nm1_c32.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_gkp_state_circuit_nq1_nm1_c32"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_gkp_state_circuit_nq1_nm1_c32"
else
    echo "Skipping strawberryfields_tf_gkp_state_circuit_nq1_nm1_c32, already completed."
fi

# Case: mrmustard_jax_gkp_state_circuit_nq1_nm1_c32
if ! grep -q '"mrmustard_jax_gkp_state_circuit_nq1_nm1_c32"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_gkp_state_circuit_nq1_nm1_c32..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload gkp_state_circuit --cutoff 32 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_gkp_state_circuit_nq1_nm1_c32.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_gkp_state_circuit_nq1_nm1_c32"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_gkp_state_circuit_nq1_nm1_c32"
else
    echo "Skipping mrmustard_jax_gkp_state_circuit_nq1_nm1_c32, already completed."
fi

# Case: strawberryfields_tf_qaoa_circuit_nq1_nm1_c16
if ! grep -q '"strawberryfields_tf_qaoa_circuit_nq1_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_qaoa_circuit_nq1_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload qaoa_circuit --cutoff 16 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_qaoa_circuit_nq1_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_qaoa_circuit_nq1_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_qaoa_circuit_nq1_nm1_c16"
else
    echo "Skipping strawberryfields_tf_qaoa_circuit_nq1_nm1_c16, already completed."
fi

# Case: mrmustard_jax_qaoa_circuit_nq1_nm1_c16
if ! grep -q '"mrmustard_jax_qaoa_circuit_nq1_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_qaoa_circuit_nq1_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload qaoa_circuit --cutoff 16 --num-modes 1 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_qaoa_circuit_nq1_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_qaoa_circuit_nq1_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_qaoa_circuit_nq1_nm1_c16"
else
    echo "Skipping mrmustard_jax_qaoa_circuit_nq1_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_qaoa_circuit_nq1_nm2_c16
if ! grep -q '"strawberryfields_tf_qaoa_circuit_nq1_nm2_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_qaoa_circuit_nq1_nm2_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload qaoa_circuit --cutoff 16 --num-modes 2 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_qaoa_circuit_nq1_nm2_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_qaoa_circuit_nq1_nm2_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_qaoa_circuit_nq1_nm2_c16"
else
    echo "Skipping strawberryfields_tf_qaoa_circuit_nq1_nm2_c16, already completed."
fi

# Case: mrmustard_jax_qaoa_circuit_nq1_nm2_c16
if ! grep -q '"mrmustard_jax_qaoa_circuit_nq1_nm2_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_qaoa_circuit_nq1_nm2_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload qaoa_circuit --cutoff 16 --num-modes 2 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_qaoa_circuit_nq1_nm2_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_qaoa_circuit_nq1_nm2_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_qaoa_circuit_nq1_nm2_c16"
else
    echo "Skipping mrmustard_jax_qaoa_circuit_nq1_nm2_c16, already completed."
fi

# Case: strawberryfields_tf_qaoa_circuit_nq1_nm4_c16
if ! grep -q '"strawberryfields_tf_qaoa_circuit_nq1_nm4_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_qaoa_circuit_nq1_nm4_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload qaoa_circuit --cutoff 16 --num-modes 4 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_qaoa_circuit_nq1_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_qaoa_circuit_nq1_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_qaoa_circuit_nq1_nm4_c16"
else
    echo "Skipping strawberryfields_tf_qaoa_circuit_nq1_nm4_c16, already completed."
fi

# Case: mrmustard_jax_qaoa_circuit_nq1_nm4_c16
if ! grep -q '"mrmustard_jax_qaoa_circuit_nq1_nm4_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_qaoa_circuit_nq1_nm4_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload qaoa_circuit --cutoff 16 --num-modes 4 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_qaoa_circuit_nq1_nm4_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_qaoa_circuit_nq1_nm4_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_qaoa_circuit_nq1_nm4_c16"
else
    echo "Skipping mrmustard_jax_qaoa_circuit_nq1_nm4_c16, already completed."
fi

# Case: strawberryfields_tf_qft_circuit_nq5_nm1_c16
if ! grep -q '"strawberryfields_tf_qft_circuit_nq5_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_qft_circuit_nq5_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload qft_circuit --cutoff 16 --num-modes 1 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_qft_circuit_nq5_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_qft_circuit_nq5_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_qft_circuit_nq5_nm1_c16"
else
    echo "Skipping strawberryfields_tf_qft_circuit_nq5_nm1_c16, already completed."
fi

# Case: mrmustard_jax_qft_circuit_nq5_nm1_c16
if ! grep -q '"mrmustard_jax_qft_circuit_nq5_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_qft_circuit_nq5_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload qft_circuit --cutoff 16 --num-modes 1 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_qft_circuit_nq5_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_qft_circuit_nq5_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_qft_circuit_nq5_nm1_c16"
else
    echo "Skipping mrmustard_jax_qft_circuit_nq5_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_qft_circuit_nq5_nm1_c32
if ! grep -q '"strawberryfields_tf_qft_circuit_nq5_nm1_c32"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_qft_circuit_nq5_nm1_c32..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload qft_circuit --cutoff 32 --num-modes 1 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_qft_circuit_nq5_nm1_c32.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_qft_circuit_nq5_nm1_c32"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_qft_circuit_nq5_nm1_c32"
else
    echo "Skipping strawberryfields_tf_qft_circuit_nq5_nm1_c32, already completed."
fi

# Case: mrmustard_jax_qft_circuit_nq5_nm1_c32
if ! grep -q '"mrmustard_jax_qft_circuit_nq5_nm1_c32"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_qft_circuit_nq5_nm1_c32..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload qft_circuit --cutoff 32 --num-modes 1 --num-qubits 5 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_qft_circuit_nq5_nm1_c32.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_qft_circuit_nq5_nm1_c32"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_qft_circuit_nq5_nm1_c32"
else
    echo "Skipping mrmustard_jax_qft_circuit_nq5_nm1_c32, already completed."
fi

# Case: strawberryfields_tf_shors_circuit_nq1_nm3_c8
if ! grep -q '"strawberryfields_tf_shors_circuit_nq1_nm3_c8"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_shors_circuit_nq1_nm3_c8..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload shors_circuit --cutoff 8 --num-modes 3 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_shors_circuit_nq1_nm3_c8.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_shors_circuit_nq1_nm3_c8"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_shors_circuit_nq1_nm3_c8"
else
    echo "Skipping strawberryfields_tf_shors_circuit_nq1_nm3_c8, already completed."
fi

# Case: mrmustard_jax_shors_circuit_nq1_nm3_c8
if ! grep -q '"mrmustard_jax_shors_circuit_nq1_nm3_c8"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_shors_circuit_nq1_nm3_c8..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload shors_circuit --cutoff 8 --num-modes 3 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_shors_circuit_nq1_nm3_c8.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_shors_circuit_nq1_nm3_c8"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_shors_circuit_nq1_nm3_c8"
else
    echo "Skipping mrmustard_jax_shors_circuit_nq1_nm3_c8, already completed."
fi

# Case: strawberryfields_tf_shors_circuit_nq1_nm3_c16
if ! grep -q '"strawberryfields_tf_shors_circuit_nq1_nm3_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_shors_circuit_nq1_nm3_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload shors_circuit --cutoff 16 --num-modes 3 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_shors_circuit_nq1_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_shors_circuit_nq1_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_shors_circuit_nq1_nm3_c16"
else
    echo "Skipping strawberryfields_tf_shors_circuit_nq1_nm3_c16, already completed."
fi

# Case: mrmustard_jax_shors_circuit_nq1_nm3_c16
if ! grep -q '"mrmustard_jax_shors_circuit_nq1_nm3_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_shors_circuit_nq1_nm3_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload shors_circuit --cutoff 16 --num-modes 3 --num-qubits 1 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_shors_circuit_nq1_nm3_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_shors_circuit_nq1_nm3_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_shors_circuit_nq1_nm3_c16"
else
    echo "Skipping mrmustard_jax_shors_circuit_nq1_nm3_c16, already completed."
fi

# Case: strawberryfields_tf_state_transfer_CVtoDV_circuit_nq2_nm1_c16
if ! grep -q '"strawberryfields_tf_state_transfer_CVtoDV_circuit_nq2_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_state_transfer_CVtoDV_circuit_nq2_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload state_transfer_CVtoDV_circuit --cutoff 16 --num-modes 1 --num-qubits 2 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_state_transfer_CVtoDV_circuit_nq2_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_state_transfer_CVtoDV_circuit_nq2_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_state_transfer_CVtoDV_circuit_nq2_nm1_c16"
else
    echo "Skipping strawberryfields_tf_state_transfer_CVtoDV_circuit_nq2_nm1_c16, already completed."
fi

# Case: mrmustard_jax_state_transfer_CVtoDV_circuit_nq2_nm1_c16
if ! grep -q '"mrmustard_jax_state_transfer_CVtoDV_circuit_nq2_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_state_transfer_CVtoDV_circuit_nq2_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload state_transfer_CVtoDV_circuit --cutoff 16 --num-modes 1 --num-qubits 2 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_state_transfer_CVtoDV_circuit_nq2_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_state_transfer_CVtoDV_circuit_nq2_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_state_transfer_CVtoDV_circuit_nq2_nm1_c16"
else
    echo "Skipping mrmustard_jax_state_transfer_CVtoDV_circuit_nq2_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_state_transfer_DVtoCV_circuit_nq2_nm1_c16
if ! grep -q '"strawberryfields_tf_state_transfer_DVtoCV_circuit_nq2_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_state_transfer_DVtoCV_circuit_nq2_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload state_transfer_DVtoCV_circuit --cutoff 16 --num-modes 1 --num-qubits 2 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_state_transfer_DVtoCV_circuit_nq2_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_state_transfer_DVtoCV_circuit_nq2_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_state_transfer_DVtoCV_circuit_nq2_nm1_c16"
else
    echo "Skipping strawberryfields_tf_state_transfer_DVtoCV_circuit_nq2_nm1_c16, already completed."
fi

# Case: mrmustard_jax_state_transfer_DVtoCV_circuit_nq2_nm1_c16
if ! grep -q '"mrmustard_jax_state_transfer_DVtoCV_circuit_nq2_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_state_transfer_DVtoCV_circuit_nq2_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload state_transfer_DVtoCV_circuit --cutoff 16 --num-modes 1 --num-qubits 2 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_state_transfer_DVtoCV_circuit_nq2_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_state_transfer_DVtoCV_circuit_nq2_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_state_transfer_DVtoCV_circuit_nq2_nm1_c16"
else
    echo "Skipping mrmustard_jax_state_transfer_DVtoCV_circuit_nq2_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_state_transfer_CVtoDV_circuit_nq4_nm1_c16
if ! grep -q '"strawberryfields_tf_state_transfer_CVtoDV_circuit_nq4_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_state_transfer_CVtoDV_circuit_nq4_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload state_transfer_CVtoDV_circuit --cutoff 16 --num-modes 1 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_state_transfer_CVtoDV_circuit_nq4_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_state_transfer_CVtoDV_circuit_nq4_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_state_transfer_CVtoDV_circuit_nq4_nm1_c16"
else
    echo "Skipping strawberryfields_tf_state_transfer_CVtoDV_circuit_nq4_nm1_c16, already completed."
fi

# Case: mrmustard_jax_state_transfer_CVtoDV_circuit_nq4_nm1_c16
if ! grep -q '"mrmustard_jax_state_transfer_CVtoDV_circuit_nq4_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_state_transfer_CVtoDV_circuit_nq4_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload state_transfer_CVtoDV_circuit --cutoff 16 --num-modes 1 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_state_transfer_CVtoDV_circuit_nq4_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_state_transfer_CVtoDV_circuit_nq4_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_state_transfer_CVtoDV_circuit_nq4_nm1_c16"
else
    echo "Skipping mrmustard_jax_state_transfer_CVtoDV_circuit_nq4_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_state_transfer_DVtoCV_circuit_nq4_nm1_c16
if ! grep -q '"strawberryfields_tf_state_transfer_DVtoCV_circuit_nq4_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_state_transfer_DVtoCV_circuit_nq4_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload state_transfer_DVtoCV_circuit --cutoff 16 --num-modes 1 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_state_transfer_DVtoCV_circuit_nq4_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_state_transfer_DVtoCV_circuit_nq4_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_state_transfer_DVtoCV_circuit_nq4_nm1_c16"
else
    echo "Skipping strawberryfields_tf_state_transfer_DVtoCV_circuit_nq4_nm1_c16, already completed."
fi

# Case: mrmustard_jax_state_transfer_DVtoCV_circuit_nq4_nm1_c16
if ! grep -q '"mrmustard_jax_state_transfer_DVtoCV_circuit_nq4_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_state_transfer_DVtoCV_circuit_nq4_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload state_transfer_DVtoCV_circuit --cutoff 16 --num-modes 1 --num-qubits 4 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_state_transfer_DVtoCV_circuit_nq4_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_state_transfer_DVtoCV_circuit_nq4_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_state_transfer_DVtoCV_circuit_nq4_nm1_c16"
else
    echo "Skipping mrmustard_jax_state_transfer_DVtoCV_circuit_nq4_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_state_transfer_CVtoDV_circuit_nq8_nm1_c16
if ! grep -q '"strawberryfields_tf_state_transfer_CVtoDV_circuit_nq8_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_state_transfer_CVtoDV_circuit_nq8_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload state_transfer_CVtoDV_circuit --cutoff 16 --num-modes 1 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_state_transfer_CVtoDV_circuit_nq8_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_state_transfer_CVtoDV_circuit_nq8_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_state_transfer_CVtoDV_circuit_nq8_nm1_c16"
else
    echo "Skipping strawberryfields_tf_state_transfer_CVtoDV_circuit_nq8_nm1_c16, already completed."
fi

# Case: mrmustard_jax_state_transfer_CVtoDV_circuit_nq8_nm1_c16
if ! grep -q '"mrmustard_jax_state_transfer_CVtoDV_circuit_nq8_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_state_transfer_CVtoDV_circuit_nq8_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload state_transfer_CVtoDV_circuit --cutoff 16 --num-modes 1 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_state_transfer_CVtoDV_circuit_nq8_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_state_transfer_CVtoDV_circuit_nq8_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_state_transfer_CVtoDV_circuit_nq8_nm1_c16"
else
    echo "Skipping mrmustard_jax_state_transfer_CVtoDV_circuit_nq8_nm1_c16, already completed."
fi

# Case: strawberryfields_tf_state_transfer_DVtoCV_circuit_nq8_nm1_c16
if ! grep -q '"strawberryfields_tf_state_transfer_DVtoCV_circuit_nq8_nm1_c16"' "$CHECKPOINT"; then
    echo "Running strawberryfields_tf_state_transfer_DVtoCV_circuit_nq8_nm1_c16..."
    .venv-sf-gpu/bin/python experiments/python/run_baseline_backend.py --backend strawberryfields_tf --workload state_transfer_DVtoCV_circuit --cutoff 16 --num-modes 1 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/strawberryfields_tf_state_transfer_DVtoCV_circuit_nq8_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("strawberryfields_tf_state_transfer_DVtoCV_circuit_nq8_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed strawberryfields_tf_state_transfer_DVtoCV_circuit_nq8_nm1_c16"
else
    echo "Skipping strawberryfields_tf_state_transfer_DVtoCV_circuit_nq8_nm1_c16, already completed."
fi

# Case: mrmustard_jax_state_transfer_DVtoCV_circuit_nq8_nm1_c16
if ! grep -q '"mrmustard_jax_state_transfer_DVtoCV_circuit_nq8_nm1_c16"' "$CHECKPOINT"; then
    echo "Running mrmustard_jax_state_transfer_DVtoCV_circuit_nq8_nm1_c16..."
    .venv-mm-gpu/bin/python experiments/python/run_baseline_backend.py --backend mrmustard_jax --workload state_transfer_DVtoCV_circuit --cutoff 16 --num-modes 1 --num-qubits 8 --warmup-runs 2 --measured-runs 5 --output experiments/results/remote-h100-baseline-sc26_baselines_hybrid_20260316/mrmustard_jax_state_transfer_DVtoCV_circuit_nq8_nm1_c16.json
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("checkpoints/sc26_baselines_hybrid_20260316.json")); d["completed"].append("mrmustard_jax_state_transfer_DVtoCV_circuit_nq8_nm1_c16"); json.dump(d, open("checkpoints/sc26_baselines_hybrid_20260316.json", "w"), indent=2)'
    echo "Completed mrmustard_jax_state_transfer_DVtoCV_circuit_nq8_nm1_c16"
else
    echo "Skipping mrmustard_jax_state_transfer_DVtoCV_circuit_nq8_nm1_c16, already completed."
fi
