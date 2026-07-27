#!/usr/bin/env python3
"""All-visible-GPU telemetry collection backed by nvidia-smi."""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from typing import Any

MAX_RETAINED_SAMPLES_PER_GPU = 512


FIELDS = (
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "pcie.link.gen.current",
    "pcie.link.width.current",
)


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def sample_gpus(gpu_ids: list[int]) -> dict[str, dict[str, float]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(FIELDS)}",
            "--format=csv,noheader,nounits",
            f"--id={','.join(map(str, gpu_ids))}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {}
    timestamp = time.time()
    samples: dict[str, dict[str, float]] = {}
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(FIELDS):
            continue
        parsed = [_number(value) for value in values]
        if parsed[0] is None:
            continue
        sample = {"timestamp_unix": timestamp}
        for field, value in zip(FIELDS[1:], parsed[1:]):
            if value is not None:
                sample[field.replace(".", "_")] = value
        samples[str(int(parsed[0]))] = sample
    return samples


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {"sample_count": 0.0}
    keys = sorted({key for sample in samples for key in sample if key != "timestamp_unix"})
    result: dict[str, float] = {"sample_count": float(len(samples))}
    for key in keys:
        values = [sample[key] for sample in samples if key in sample]
        if values:
            result[f"avg_{key}"] = statistics.fmean(values)
            result[f"peak_{key}"] = max(values)
    energy = 0.0
    for previous, current in zip(samples, samples[1:]):
        if "power_draw" in previous and "power_draw" in current:
            duration = max(0.0, current["timestamp_unix"] - previous["timestamp_unix"])
            energy += duration * (previous["power_draw"] + current["power_draw"]) / 2.0
    result["energy_joules"] = energy
    return result


class TelemetryCollector:
    def __init__(self, gpu_ids: list[int], interval_ms: int) -> None:
        self.gpu_ids = gpu_ids
        # Forking nvidia-smi at 10 Hz measurably perturbs short CUDA workloads
        # and creates multi-million-line result files for timeout cases.
        self.interval_s = max(1.0, interval_ms / 1000.0)
        self.samples: dict[str, list[dict[str, float]]] = {str(gpu): [] for gpu in gpu_ids}
        self._summary_state: dict[str, dict[str, Any]] = {
            str(gpu): {
                "sample_count": 0,
                "sums": {},
                "peaks": {},
                "energy_joules": 0.0,
                "previous": None,
            }
            for gpu in gpu_ids
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _record(self, gpu: str, sample: dict[str, float]) -> None:
        state = self._summary_state.setdefault(
            gpu,
            {
                "sample_count": 0,
                "sums": {},
                "peaks": {},
                "energy_joules": 0.0,
                "previous": None,
            },
        )
        state["sample_count"] += 1
        for key, value in sample.items():
            if key == "timestamp_unix":
                continue
            state["sums"][key] = state["sums"].get(key, 0.0) + value
            state["peaks"][key] = max(state["peaks"].get(key, value), value)
        previous = state["previous"]
        if previous and "power_draw" in previous and "power_draw" in sample:
            duration = max(0.0, sample["timestamp_unix"] - previous["timestamp_unix"])
            state["energy_joules"] += (
                duration * (previous["power_draw"] + sample["power_draw"]) / 2.0
            )
        state["previous"] = sample

        retained = self.samples.setdefault(gpu, [])
        count = state["sample_count"]
        if len(retained) < MAX_RETAINED_SAMPLES_PER_GPU:
            retained.append(sample)
        else:
            # Deterministic reservoir sampling keeps the artifact bounded while
            # retaining coverage across the complete run.
            replacement = (count * 2654435761) % count
            if replacement < MAX_RETAINED_SAMPLES_PER_GPU:
                retained[replacement] = sample

    @staticmethod
    def _online_summary(state: dict[str, Any]) -> dict[str, float]:
        count = int(state["sample_count"])
        if count <= 0:
            return {"sample_count": 0.0}
        result: dict[str, float] = {"sample_count": float(count)}
        for key, total in state["sums"].items():
            result[f"avg_{key}"] = total / count
            result[f"peak_{key}"] = state["peaks"][key]
        result["energy_joules"] = state["energy_joules"]
        return result

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            for gpu, sample in sample_gpus(self.gpu_ids).items():
                self._record(gpu, sample)
            self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(5.0, self.interval_s * 2.0))
        for gpu, sample in sample_gpus(self.gpu_ids).items():
            self._record(gpu, sample)
        per_gpu = {
            gpu: {
                "samples": samples,
                "samples_retained": len(samples),
                "samples_truncated": self._summary_state[gpu]["sample_count"] > len(samples),
                "summary": self._online_summary(self._summary_state[gpu]),
            }
            for gpu, samples in self.samples.items()
        }
        avg_utils = [
            value["summary"].get("avg_utilization_gpu", 0.0)
            for value in per_gpu.values()
            if value["summary"].get("sample_count", 0) > 0
        ]
        aggregate = {
            "sample_count": sum(value["summary"].get("sample_count", 0) for value in per_gpu.values()),
            "energy_joules": sum(value["summary"].get("energy_joules", 0) for value in per_gpu.values()),
            "gpu_util_imbalance_pct": max(avg_utils) - min(avg_utils) if avg_utils else 0.0,
        }
        return {"per_gpu": per_gpu, "aggregate": aggregate}
