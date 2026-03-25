import json
import os

with open('experiments/configs/sc26_scaling.json', 'r') as f:
    config = json.load(f)

run_id = "sc26_baselines_hybrid_20260316"
remote_root = "/home/xiangdebin/sandboxes/HybridCVDV-Simulator"
output_dir = f"experiments/results/remote-h100-baseline-{run_id}"
checkpoint_path = f"checkpoints/{run_id}.json"

commands = []
seen_specs = set()

for case in config["cases"]:
    workload = case["workload"]
    num_qubits = case.get("num_qubits", 0)
    num_modes = case.get("num_modes", 0)
    cutoff = case.get("cutoff", 16)
    
    spec = (workload, cutoff, num_modes, num_qubits)
    if spec in seen_specs:
        continue
    seen_specs.add(spec)

    for backend in ["strawberryfields_tf", "mrmustard_jax"]:
        python_bin = config["backend_python_map"][backend]
        
        cmd = [
            f"{python_bin} experiments/python/run_baseline_backend.py",
            f"--backend {backend}",
            f"--workload {workload}",
            f"--cutoff {cutoff}",
            f"--num-modes {num_modes}",
            f"--num-qubits {num_qubits}",
            f"--warmup-runs {case.get('warmup_runs', 2)}",
            f"--measured-runs {case.get('measured_runs', 5)}",
        ]
        
        if workload == "vqe_circuit":
            cmd.append(f"--layers {case.get('layers', 2)}")
        elif workload == "jch_simulation_circuit":
            cmd.append(f"--timesteps {case.get('timesteps', 5)}")
            
        case_id = f"{backend}_{workload}_nq{num_qubits}_nm{num_modes}_c{cutoff}"
        cmd.append(f"--output {output_dir}/{case_id}.json")
        
        commands.append({"id": case_id, "command": " ".join(cmd)})

# Generate run_baselines.sh
run_script = f"""#!/bin/bash
set -e
cd {remote_root}
mkdir -p {output_dir}
mkdir -p checkpoints

CHECKPOINT="{checkpoint_path}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "{{\\"completed\\": []}}" > "$CHECKPOINT"
fi

"""

for cmd in commands:
    run_script += f"""
# Case: {cmd['id']}
if ! grep -q '"{cmd['id']}"' "$CHECKPOINT"; then
    echo "Running {cmd['id']}..."
    {cmd['command']}
    # Update checkpoint
    python3 -c 'import json, sys; d=json.load(open("{checkpoint_path}")); d["completed"].append("{cmd["id"]}"); json.dump(d, open("{checkpoint_path}", "w"), indent=2)'
    echo "Completed {cmd['id']}"
else
    echo "Skipping {cmd['id']}, already completed."
fi
"""

with open('run_baselines_remote.sh', 'w') as f:
    f.write(run_script)

with open('resume_baselines_remote.sh', 'w') as f:
    f.write(run_script)

print(f"Generated scripts for {len(commands)} baseline runs (including hybrid checks).")
