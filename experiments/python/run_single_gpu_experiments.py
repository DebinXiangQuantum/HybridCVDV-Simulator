#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "single_gpu_experiments.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BASELINE_PYTHON = REPO_ROOT / "baselines" / "venv" / "bin" / "python"
MAX_LOG_CHARS = 4000


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def choose_baseline_python(explicit_python: str | None) -> str:
    if explicit_python:
        return explicit_python
    if DEFAULT_BASELINE_PYTHON.exists():
        return str(DEFAULT_BASELINE_PYTHON)
    return sys.executable


def run_command(cmd: list[str], cwd: pathlib.Path) -> tuple[int, str, str]:
    completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return completed.returncode, completed.stdout, completed.stderr


def truncate_text(text: str, max_chars: int = MAX_LOG_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    keep = max_chars - 64
    return text[:keep] + "\n...[truncated]...\n", True


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate single-GPU HybridCVDV experiments and write JSON artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--baseline-python", default=None)
    parser.add_argument("--internal-suite", default=None, help="Override the internal suite from config, e.g. correctness or microbench.")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    config = read_json(pathlib.Path(args.config))
    build_dir = pathlib.Path(args.build_dir)
    results_root = pathlib.Path(args.results_dir)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": iso_now(),
        "run_id": run_id,
        "single_gpu_only": True,
        "config_path": str(pathlib.Path(args.config).resolve()),
        "artifacts": [],
    }

    binary_name = config.get("internal_binary", "hybridcvdv_single_gpu_experiments")
    internal_binary = build_dir / binary_name
    internal_output = run_dir / config.get("internal_output", "internal_single_gpu.json")
    internal_suite = args.internal_suite or config.get("internal_suite", "all")
    gaussian_symbolic_mode_limit = config.get("gaussian_symbolic_mode_limit", 16)

    if internal_binary.exists():
        cmd = [
            str(internal_binary),
            "--suite",
            internal_suite,
            "--gaussian-symbolic-mode-limit",
            str(gaussian_symbolic_mode_limit),
            "--output",
            str(internal_output),
        ]
        returncode, stdout, stderr = run_command(cmd, REPO_ROOT)
        stdout_text, stdout_truncated = truncate_text(stdout)
        stderr_text, stderr_truncated = truncate_text(stderr)
        manifest["artifacts"].append(
            {
                "kind": "internal",
                "path": str(internal_output),
                "status": "ok" if returncode == 0 else "error",
                "command": cmd,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "returncode": returncode,
            }
        )
    else:
        manifest["artifacts"].append(
            {
                "kind": "internal",
                "path": str(internal_output),
                "status": "missing_binary",
                "command": [],
                "stdout": "",
                "stderr": f"expected internal experiment binary at {internal_binary}",
                "returncode": None,
            }
        )

    if not args.skip_baselines:
        baseline_python = choose_baseline_python(args.baseline_python)
        baseline_runner = REPO_ROOT / "experiments" / "python" / "run_baseline_backend.py"
        for backend in config.get("baseline_backends", []):
            output_path = run_dir / f"baseline_{backend}.json"
            cmd = [baseline_python, str(baseline_runner), "--backend", backend, "--output", str(output_path)]
            returncode, stdout, stderr = run_command(cmd, REPO_ROOT)
            stdout_text, stdout_truncated = truncate_text(stdout)
            stderr_text, stderr_truncated = truncate_text(stderr)
            manifest["artifacts"].append(
                {
                    "kind": "baseline",
                    "backend": backend,
                    "path": str(output_path),
                    "status": "ok" if returncode == 0 else "error",
                    "command": cmd,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "returncode": returncode,
                }
            )

    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote experiment manifest to {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
