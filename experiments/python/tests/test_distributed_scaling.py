from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from distributed_common import atomic_write_json, classify_failure, empty_result, validate_result
from generate_distributed_config import generate
from generate_sc26_baseline_qasm import render_qasm
from merge_distributed_scaling import merge
from distributed_telemetry import MAX_RETAINED_SAMPLES_PER_GPU, TelemetryCollector
from run_distributed_scaling import (
    filter_gpu_scaling_eligible_cases,
    is_known_host_bound_case,
    successful_work_from_manifest,
)


class DistributedConfigTest(unittest.TestCase):
    def test_derives_missing_cutoff_without_duplicates(self) -> None:
        source = {
            "schema_version": "2.0",
            "cases": [
                {
                    "name": "sc26_qft_nq5_c16",
                    "workload": "qft_circuit",
                    "cutoff": 16,
                    "num_modes": 1,
                    "num_qubits": 5,
                    "internal_name_filter": "sc26_qft_nq5_c16",
                    "backends": ["hybridcvdv"],
                }
            ],
        }
        config, summary = generate(source, [4, 8, 16, 32])
        self.assertEqual(summary["output_case_count"], 4)
        self.assertEqual(len({case["name"] for case in config["cases"]}), 4)
        self.assertEqual({case["cutoff"] for case in config["cases"]}, {4, 8, 16, 32})

    def test_qasm_is_deterministic_and_uses_supported_subset(self) -> None:
        case = {
            "name": "sc26_vqe_nq2_nm1_c8",
            "workload": "vqe_circuit",
            "cutoff": 8,
            "num_modes": 1,
            "num_qubits": 2,
            "layers": 2,
        }
        first, qubits, gates = render_qasm(case)
        second, _, _ = render_qasm(case)
        self.assertEqual(first, second)
        self.assertEqual(qubits, 5)
        self.assertGreater(gates, 0)
        self.assertNotIn("measure", first)
        self.assertNotIn("rx(", first)


class SchemaTest(unittest.TestCase):
    def test_failure_classification(self) -> None:
        self.assertEqual(classify_failure(1, "", "CUDA illegal memory access"), "crash_cuda")
        self.assertEqual(classify_failure(1, "", "out of memory"), "oom_single_gpu_pool")
        self.assertEqual(classify_failure(None, "", "", timed_out=True), "timeout")

    def test_result_validation(self) -> None:
        payload = empty_result(
            system="hybridcvdv",
            case={"name": "case", "cutoff": 4, "num_modes": 1, "num_qubits": 1},
            phase="smoke",
            gpu_ids=[0, 1],
            warmup_runs=0,
            measured_runs=1,
            repetition=0,
        )
        payload["status"] = "ok"
        self.assertEqual(validate_result(payload), [])

    def test_telemetry_retention_is_bounded_with_exact_summary(self) -> None:
        collector = TelemetryCollector([0], 100)
        for index in range(MAX_RETAINED_SAMPLES_PER_GPU + 100):
            collector._record(
                "0",
                {
                    "timestamp_unix": float(index),
                    "utilization_gpu": float(index % 100),
                    "power_draw": 100.0,
                },
            )
        state = collector._summary_state["0"]
        summary = collector._online_summary(state)
        self.assertEqual(len(collector.samples["0"]), MAX_RETAINED_SAMPLES_PER_GPU)
        self.assertEqual(
            summary["sample_count"], float(MAX_RETAINED_SAMPLES_PER_GPU + 100)
        )
        self.assertEqual(summary["peak_utilization_gpu"], 99.0)

    def test_strong_eligibility_manifest_filters_host_bound_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            eligible_path = root / "eligible.json"
            host_path = root / "host.json"
            atomic_write_json(
                eligible_path,
                {"diagnostics": {"gpu_scaling_eligible": True}},
            )
            atomic_write_json(
                host_path,
                {"diagnostics": {"gpu_scaling_eligible": False}},
            )
            manifest_path = root / "manifest.json"
            atomic_write_json(
                manifest_path,
                {
                    "artifacts": [
                        {
                            "system": "hybridcvdv",
                            "phase": "strong",
                            "gpu_count": 1,
                            "status": "ok",
                            "case_name": "eligible",
                            "path": str(eligible_path),
                        },
                        {
                            "system": "hybridcvdv",
                            "phase": "strong",
                            "gpu_count": 1,
                            "status": "ok",
                            "case_name": "host",
                            "path": str(host_path),
                        },
                    ]
                },
            )
            cases = [{"name": "eligible"}, {"name": "host"}]
            self.assertEqual(
                filter_gpu_scaling_eligible_cases(
                    cases, manifest_path, "hybridcvdv"
                ),
                [{"name": "eligible"}],
            )

    def test_known_host_bound_nq16_transfer_is_skipped(self) -> None:
        self.assertTrue(
            is_known_host_bound_case(
                "hybridcvdv", "sc26_transfer_DVtoCV_nq16_c4"
            )
        )
        self.assertFalse(
            is_known_host_bound_case(
                "hybridcvdv", "sc26_transfer_DVtoCV_nq4_c4"
            )
        )
        self.assertFalse(
            is_known_host_bound_case("atlas", "sc26_transfer_DVtoCV_nq16_c4")
        )

    def test_successful_work_manifest_filters_exact_gpu_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = pathlib.Path(directory) / "manifest.json"
            atomic_write_json(
                manifest_path,
                {
                    "artifacts": [
                        {
                            "system": "atlas",
                            "phase": "full",
                            "case_name": "case",
                            "gpu_count": 1,
                            "status": "ok",
                        },
                        {
                            "system": "atlas",
                            "phase": "full",
                            "case_name": "case",
                            "gpu_count": 2,
                            "status": "oom_single_gpu_pool",
                        },
                    ]
                },
            )
            self.assertEqual(
                successful_work_from_manifest(manifest_path),
                {("atlas", "full", "case", 1)},
            )


class MergeTest(unittest.TestCase):
    def test_strong_speedup_requires_one_gpu_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for gpu_count, simulation_ms in ((1, 10.0), (2, 6.0)):
                path = root / "hybridcvdv/strong" / f"g{gpu_count}" / "case" / "r0/result.json"
                payload = empty_result(
                    system="hybridcvdv",
                    case={"name": "case", "cutoff": 4, "num_modes": 1, "num_qubits": 1},
                    phase="strong",
                    gpu_ids=list(range(gpu_count)),
                    warmup_runs=0,
                    measured_runs=1,
                    repetition=0,
                )
                payload["status"] = "ok"
                payload["timing"]["simulation_ms"] = simulation_ms
                atomic_write_json(path, payload)
            results, failures = merge(root, root / "merged")
            self.assertFalse(failures)
            two_gpu = next(result for result in results if result["gpu_count"] == 2)
            self.assertAlmostEqual(two_gpu["scaling"]["speedup"], 10.0 / 6.0)

    def test_checksum_mismatch_is_classified_as_incorrect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for gpu_count, checksum in ((1, 1.0), (2, 1.1)):
                path = root / "atlas/strong" / f"g{gpu_count}" / "case" / "r0/result.json"
                payload = empty_result(
                    system="atlas",
                    case={"name": "case", "cutoff": 4, "num_modes": 1, "num_qubits": 3},
                    phase="strong",
                    gpu_ids=list(range(gpu_count)),
                    warmup_runs=0,
                    measured_runs=1,
                    repetition=0,
                )
                payload["status"] = "ok"
                payload["correctness"]["checksum"] = checksum
                atomic_write_json(path, payload)
            results, failures = merge(root, root / "merged")
            two_gpu = next(result for result in results if result["gpu_count"] == 2)
            self.assertEqual(two_gpu["status"], "incorrect_result")
            self.assertEqual(len(failures), 1)

    def test_host_bound_control_has_no_gpu_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for gpu_count, simulation_ms in ((1, 10.0), (2, 20.0)):
                path = (
                    root
                    / "hybridcvdv/strong"
                    / f"g{gpu_count}"
                    / "control"
                    / "r0/result.json"
                )
                payload = empty_result(
                    system="hybridcvdv",
                    case={
                        "name": "control",
                        "cutoff": 4,
                        "num_modes": 1,
                        "num_qubits": 4,
                    },
                    phase="strong",
                    gpu_ids=list(range(gpu_count)),
                    warmup_runs=0,
                    measured_runs=1,
                    repetition=0,
                )
                payload["status"] = "ok"
                payload["timing"]["simulation_ms"] = simulation_ms
                payload["diagnostics"] = {
                    "gpu_scaling_eligible": False,
                    "scaling_role": "host_bound_control",
                }
                atomic_write_json(path, payload)
            results, failures = merge(root, root / "merged")
            self.assertFalse(failures)
            self.assertTrue(
                all("scaling" not in result for result in results)
            )


if __name__ == "__main__":
    unittest.main()
