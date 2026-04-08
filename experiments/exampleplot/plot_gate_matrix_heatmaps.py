#!/usr/bin/env python3
"""Plot example heatmaps for Level 2 single-mode gates and BS Level 3 sectors.

This script mirrors the current implementation choices in the simulator:

- Level 2 single-mode gates:
  - Displacement is shown as a dense Fock-basis matrix.
  - Squeezing is shown both as a dense Fock-basis matrix and as the current
    ELL value storage used by the cached GPU path.

- Level 3 two-mode beam splitter:
  - A dense D^2 x D^2 matrix is shown in lexicographic |m,n> / |p,q> order.
  - The cached photon-number sector submatrices U^(k) are shown separately,
    matching the packed storage logic used in the current BS path.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.configs.paper_style import (
    DOUBLE_COLUMN_PT,
    SINGLE_COLUMN_PT,
    apply_paper_style,
)


PALETTE_LIGHT = "#FFDDAB"
PALETTE_DARK = "#945034"
ELL_THRESHOLD = 1e-12


def build_colormap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "amber_brown_gate",
        [PALETTE_LIGHT, PALETTE_DARK],
    )
    cmap.set_bad("#FFF8EE")
    return cmap


def matrix_exponential(matrix: np.ndarray) -> np.ndarray:
    """Scaling-and-squaring Taylor exponential, matching the project style."""
    dim = matrix.shape[0]
    norm = float(np.max(np.sum(np.abs(matrix), axis=0)))
    scaling_power = int(math.ceil(math.log2(norm))) if norm > 1.0 else 0
    scale = float(2**scaling_power)
    scaled = matrix / scale

    result = np.eye(dim, dtype=np.complex128)
    term = np.eye(dim, dtype=np.complex128)

    for order in range(1, 81):
        term = (term @ scaled) / float(order)
        result = result + term
        if float(np.max(np.abs(term))) < 1e-14:
            break

    for _ in range(scaling_power):
        result = result @ result

    return result


def build_displacement_matrix(cutoff: int, alpha: complex) -> np.ndarray:
    annihilation = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for n in range(1, cutoff):
        annihilation[n - 1, n] = math.sqrt(float(n))
    creation = annihilation.conj().T
    generator = alpha * creation - np.conj(alpha) * annihilation
    return matrix_exponential(generator)


def build_squeezing_matrix(cutoff: int, r: float, theta: float) -> np.ndarray:
    sqrt_n = np.sqrt(np.arange(cutoff, dtype=np.float64))
    eitheta_tanhr = np.exp(1j * theta) * math.tanh(r)
    sechr = 1.0 / math.cosh(r)

    r00 = -eitheta_tanhr
    r01 = sechr
    r11 = np.conj(eitheta_tanhr)

    matrix = np.zeros((cutoff, cutoff), dtype=np.complex128)
    matrix[0, 0] = math.sqrt(sechr)

    for m in range(2, cutoff, 2):
        matrix[m, 0] = sqrt_n[m - 1] / sqrt_n[m] * r00 * matrix[m - 2, 0]

    for m in range(cutoff):
        for n in range(1, cutoff):
            if (m + n) % 2 != 0:
                continue

            term1 = 0.0j
            term2 = 0.0j
            if n >= 2:
                term1 = sqrt_n[n - 1] / sqrt_n[n] * r11 * matrix[m, n - 2]
            if m >= 1:
                term2 = sqrt_n[m] / sqrt_n[n] * r01 * matrix[m - 1, n - 1]
            matrix[m, n] = term1 + term2

    return matrix


def build_phase_rotation_matrix(cutoff: int, theta: float) -> np.ndarray:
    n = np.arange(cutoff, dtype=np.float64)
    return np.diag(np.exp(-1j * theta * n))


def build_kerr_matrix(cutoff: int, chi: float) -> np.ndarray:
    n = np.arange(cutoff, dtype=np.float64)
    return np.diag(np.exp(1j * chi * n * n))


def build_creation_matrix(cutoff: int) -> np.ndarray:
    matrix = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for n in range(1, cutoff):
        matrix[n, n - 1] = math.sqrt(float(n))
    return matrix


def build_annihilation_matrix(cutoff: int) -> np.ndarray:
    matrix = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for n in range(cutoff - 1):
        matrix[n, n + 1] = math.sqrt(float(n + 1))
    return matrix


def convert_to_ell_storage(matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    cutoff = matrix.shape[0]
    ell_values = np.full((cutoff, cutoff), np.nan + 0.0j, dtype=np.complex128)
    ell_cols = np.full((cutoff, cutoff), -1, dtype=np.int32)

    for row in range(cutoff):
        slot = 0
        for col in range(cutoff):
            value = matrix[row, col]
            if abs(value) > threshold:
                ell_values[row, slot] = value
                ell_cols[row, slot] = col
                slot += 1

    return ell_values, ell_cols


def build_bs_tensor_recursive(cutoff: int, theta: float, phi: float) -> np.ndarray:
    tensor = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=np.complex128)

    ct = math.cos(theta)
    st = math.sin(theta)
    phase = np.exp(1j * phi)
    sqrt_table = np.sqrt(np.arange(cutoff, dtype=np.float64))

    tensor[0, 0, 0, 0] = 1.0 + 0.0j

    for m in range(cutoff):
        for n in range(cutoff - m):
            p = m + n
            if 0 < p < cutoff:
                acc = 0.0j
                if m > 0:
                    acc += ct * sqrt_table[m] / sqrt_table[p] * tensor[m - 1, n, p - 1, 0]
                if n > 0:
                    acc += st * phase * sqrt_table[n] / sqrt_table[p] * tensor[m, n - 1, p - 1, 0]
                tensor[m, n, p, 0] = acc

    for m in range(cutoff):
        for n in range(cutoff):
            for p in range(cutoff):
                q = m + n - p
                if 0 < q < cutoff:
                    acc = 0.0j
                    if m > 0:
                        acc += (
                            -st
                            * np.conj(phase)
                            * sqrt_table[m]
                            / sqrt_table[q]
                            * tensor[m - 1, n, p, q - 1]
                        )
                    if n > 0:
                        acc += ct * sqrt_table[n] / sqrt_table[q] * tensor[m, n - 1, p, q - 1]
                    tensor[m, n, p, q] = acc

    return tensor


def build_bs_dense_matrix(cutoff: int, theta: float, phi: float) -> np.ndarray:
    tensor = build_bs_tensor_recursive(cutoff, theta, phi)
    return tensor.reshape(cutoff * cutoff, cutoff * cutoff)


def bs_sector_size(cutoff: int, total_photons: int) -> int:
    lower = max(0, total_photons - (cutoff - 1))
    upper = min(cutoff - 1, total_photons)
    return max(0, upper - lower + 1)


def sector_order_indices(cutoff: int) -> tuple[list[int], list[int]]:
    ordered_indices: list[int] = []
    separators: list[int] = []
    running = 0

    for k in range(2 * cutoff - 1):
        sector_pairs: list[tuple[int, int]] = []
        for mode2 in range(cutoff):
            mode1 = k - mode2
            if 0 <= mode1 < cutoff:
                sector_pairs.append((mode1, mode2))
        if not sector_pairs:
            continue
        ordered_indices.extend([mode1 * cutoff + mode2 for mode1, mode2 in sector_pairs])
        running += len(sector_pairs)
        if k < 2 * cutoff - 2:
            separators.append(running)

    return ordered_indices, separators[:-1] if separators else separators


def reorder_bs_matrix_by_sector(dense: np.ndarray, cutoff: int) -> tuple[np.ndarray, list[int]]:
    ordered_indices, separators = sector_order_indices(cutoff)
    reordered = dense[np.ix_(ordered_indices, ordered_indices)]
    return reordered, separators


def build_bs_subspace_matrices(cutoff: int, theta: float, phi: float) -> list[np.ndarray]:
    tensor = build_bs_tensor_recursive(cutoff, theta, phi)
    matrices: list[np.ndarray] = []

    for k in range(2 * cutoff - 1):
        sub_dim = k + 1
        sub = np.zeros((sub_dim, sub_dim), dtype=np.complex128)
        for out_mode2 in range(sub_dim):
            out_mode1 = k - out_mode2
            for input_mode2 in range(sub_dim):
                input_mode1 = k - input_mode2
                if (
                    out_mode1 < cutoff
                    and out_mode2 < cutoff
                    and input_mode1 < cutoff
                    and input_mode2 < cutoff
                ):
                    sub[out_mode2, input_mode2] = tensor[
                        out_mode1, out_mode2, input_mode1, input_mode2
                    ]
        matrices.append(sub)

    return matrices


def subspace_offset(k: int) -> int:
    return k * (k + 1) * (2 * k + 1) // 6


def normalize_state(state: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        return state
    return state / norm


def build_single_mode_example_state(cutoff: int) -> np.ndarray:
    state = np.zeros(cutoff, dtype=np.complex128)
    examples = [
        (0, 0.78 + 0.00j),
        (2, 0.46 + 0.12j),
        (4, -0.31 + 0.18j),
        (6, 0.19 - 0.11j),
    ]
    for idx, value in examples:
        if idx < cutoff:
            state[idx] = value
    return normalize_state(state)


def build_single_mode_phase_example_state(cutoff: int) -> np.ndarray:
    state = np.zeros(cutoff, dtype=np.complex128)
    for n in range(cutoff):
        magnitude = 0.35 + 0.10 * (1.0 + math.sin(0.75 * float(n))) + 0.07 * (n + 1)
        phase = 0.42 * float(n) - 0.11 * float(n * n)
        state[n] = magnitude * np.exp(1j * phase)
    return normalize_state(state)


def build_two_qumode_parallel_example_state(cutoff: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    amplitudes = rng.uniform(0.10, 1.00, size=(cutoff, cutoff))
    phases = rng.uniform(-math.pi, math.pi, size=(cutoff, cutoff))
    state = amplitudes * np.exp(1j * phases)
    return normalize_state(state)


def apply_single_mode_matrix_on_last_mode(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return state @ gate.T


def apply_single_mode_matrix_on_first_mode(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return gate @ state


def build_two_mode_example_state(cutoff: int) -> np.ndarray:
    state = np.zeros((cutoff, cutoff), dtype=np.complex128)
    examples = [
        ((2, 0), 0.62 + 0.00j),
        ((1, 1), 0.43 + 0.08j),
        ((0, 2), 0.35 - 0.10j),
        ((3, 1), 0.24 + 0.05j),
    ]
    for (mode1, mode2), value in examples:
        if mode1 < cutoff and mode2 < cutoff:
            state[mode1, mode2] = value
    return normalize_state(state)


def lift_single_mode_sparse_operator_to_two_mode(local_sparse: np.ndarray, cutoff: int) -> np.ndarray:
    full_dim = cutoff * cutoff
    lifted = np.full((full_dim, full_dim), np.nan + 0.0j, dtype=np.complex128)
    for spectator in range(cutoff):
        start = spectator * cutoff
        lifted[start : start + cutoff, start : start + cutoff] = local_sparse
    return lifted


def extract_sector_vector(state: np.ndarray, total_photons: int) -> np.ndarray:
    cutoff = state.shape[0]
    values: list[complex] = []
    for mode2 in range(total_photons + 1):
        mode1 = total_photons - mode2
        if 0 <= mode1 < cutoff and 0 <= mode2 < cutoff:
            values.append(state[mode1, mode2])
    return np.asarray(values, dtype=np.complex128)


def embed_sector_vector(cutoff: int, total_photons: int, vector: np.ndarray) -> np.ndarray:
    state = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for mode2, value in enumerate(vector):
        mode1 = total_photons - mode2
        if 0 <= mode1 < cutoff and 0 <= mode2 < cutoff:
            state[mode1, mode2] = value
    return state


def build_bs_sector_example_state(cutoff: int, total_photons: int) -> np.ndarray:
    sector_dim = bs_sector_size(cutoff, total_photons)
    vector = np.zeros(sector_dim, dtype=np.complex128)
    base = [
        0.55 + 0.00j,
        0.42 + 0.12j,
        0.31 - 0.08j,
        0.19 + 0.06j,
        0.11 - 0.03j,
    ]
    for idx in range(min(sector_dim, len(base))):
        vector[idx] = base[idx]
    vector = normalize_state(vector)
    return embed_sector_vector(cutoff, total_photons, vector)


def build_bs_multi_sector_example_state(cutoff: int, sectors: list[int]) -> np.ndarray:
    rng = np.random.default_rng(7)
    amplitudes = rng.uniform(0.08, 0.55, size=(cutoff, cutoff))
    phases = rng.uniform(-math.pi, math.pi, size=(cutoff, cutoff))
    state = amplitudes * np.exp(1j * phases)

    sector_weights = [1.65, 1.40, 1.58]
    for idx, total_photons in enumerate(sectors):
        weight = sector_weights[min(idx, len(sector_weights) - 1)]
        state += weight * build_bs_sector_example_state(cutoff, total_photons)

    return normalize_state(state)


def draw_square_lattice(ax: plt.Axes, cutoff: int) -> None:
    boundaries = np.arange(-0.5, cutoff, 1.0)
    ax.set_xticks(np.arange(cutoff))
    ax.set_yticks(np.arange(cutoff))
    ax.set_xticks(boundaries, minor=True)
    ax.set_yticks(boundaries, minor=True)
    ax.grid(which="minor", color="#7A5834", linewidth=0.35, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_rect_lattice(ax: plt.Axes, rows: int, cols: int) -> None:
    x_boundaries = np.arange(-0.5, cols, 1.0)
    y_boundaries = np.arange(-0.5, rows, 1.0)
    ax.set_xticks(x_boundaries, minor=True)
    ax.set_yticks(y_boundaries, minor=True)
    ax.grid(which="minor", color="#7A5834", linewidth=0.35, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)


def apply_two_mode_tensor_to_state(tensor: np.ndarray, state: np.ndarray) -> np.ndarray:
    cutoff = state.shape[0]
    output = np.zeros_like(state)
    for out_mode1 in range(cutoff):
        for out_mode2 in range(cutoff):
            total = 0.0j
            for in_mode1 in range(cutoff):
                for in_mode2 in range(cutoff):
                    total += tensor[out_mode1, out_mode2, in_mode1, in_mode2] * state[in_mode1, in_mode2]
            output[out_mode1, out_mode2] = total
    return output


def render_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    cmap: LinearSegmentedColormap,
    title: str,
    xlabel: str,
    ylabel: str,
    vmax: float | None = None,
    add_separators: list[int] | None = None,
    mask_invalid: bool = False,
    aspect: str | float = "equal",
):
    magnitude = np.abs(values)
    if mask_invalid:
        magnitude = np.ma.masked_invalid(magnitude)

    image = ax.imshow(
        magnitude,
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect=aspect,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    if add_separators:
        for pos in add_separators:
            ax.axhline(pos - 0.5, color="#7A5834", linewidth=0.45, alpha=0.55)
            ax.axvline(pos - 0.5, color="#7A5834", linewidth=0.45, alpha=0.55)

    return image


def render_scalar_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    cmap: LinearSegmentedColormap,
    title: str,
    xlabel: str,
    ylabel: str,
    vmin: float,
    vmax: float,
    mask_invalid: bool = False,
    aspect: str | float = "equal",
):
    display = values
    if mask_invalid:
        display = np.ma.masked_invalid(values)

    image = ax.imshow(
        display,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect=aspect,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    return image


def save_svg(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def wrap_phase_over_pi(values: np.ndarray) -> np.ndarray:
    return np.mod(np.angle(values), 2.0 * math.pi) / math.pi


def diagonal_phase_display(matrix: np.ndarray) -> np.ndarray:
    cutoff = matrix.shape[0]
    display = np.full((cutoff, cutoff), np.nan, dtype=np.float64)
    diag_values = wrap_phase_over_pi(np.diag(matrix))
    for idx, value in enumerate(diag_values):
        display[idx, idx] = value
    return display


def level1_storage_vector(kind: str, cutoff: int) -> np.ndarray:
    values = np.zeros(cutoff, dtype=np.float64)
    if kind == "creation":
        for n in range(1, cutoff):
            values[n] = math.sqrt(float(n))
        return values
    if kind == "annihilation":
        for n in range(cutoff - 1):
            values[n] = math.sqrt(float(n + 1))
        return values
    raise ValueError(f"unknown level1 storage kind: {kind}")


def plot_level0_single_mode_heatmaps(
    output_dir: Path,
    cutoff: int,
    theta: float,
    chi: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    phase_rotation = build_phase_rotation_matrix(cutoff, theta)
    kerr = build_kerr_matrix(cutoff, chi)

    phase_dense = diagonal_phase_display(phase_rotation)
    kerr_dense = diagonal_phase_display(kerr)
    phase_storage = wrap_phase_over_pi(np.diag(phase_rotation))[np.newaxis, :]
    kerr_storage = wrap_phase_over_pi(np.diag(kerr))[np.newaxis, :]

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=2,
        nrows=2,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.ravel()

    render_scalar_heatmap(
        axes_flat[0],
        phase_dense,
        cmap=cmap,
        title=rf"Phase rotation diagonal $R(\theta)$, $\theta={theta:.2f}$",
        xlabel="Input Fock index $n$",
        ylabel="Output Fock index $m$",
        vmin=0.0,
        vmax=2.0,
        mask_invalid=True,
    )
    render_scalar_heatmap(
        axes_flat[1],
        phase_storage,
        cmap=cmap,
        title=r"Phase storage vector $\phi[n]/\pi$",
        xlabel="Fock index $n$",
        ylabel="Stored diagonal",
        vmin=0.0,
        vmax=2.0,
        aspect="auto",
    )
    render_scalar_heatmap(
        axes_flat[2],
        kerr_dense,
        cmap=cmap,
        title=rf"Kerr diagonal $K(\chi)$, $\chi={chi:.2f}$",
        xlabel="Input Fock index $n$",
        ylabel="Output Fock index $m$",
        vmin=0.0,
        vmax=2.0,
        mask_invalid=True,
    )
    image = render_scalar_heatmap(
        axes_flat[3],
        kerr_storage,
        cmap=cmap,
        title=r"Kerr storage vector $\phi[n]/\pi$",
        xlabel="Fock index $n$",
        ylabel="Stored diagonal",
        vmin=0.0,
        vmax=2.0,
        aspect="auto",
    )

    fig.colorbar(image, ax=axes_flat.tolist(), fraction=0.022, pad=0.02, label=r"Wrapped phase / $\pi$")
    output_path = output_dir / "level0_single_mode_heatmaps.svg"
    save_svg(fig, output_path)
    return output_path


def plot_level0_single_mode_flow(
    output_dir: Path,
    cutoff: int,
    theta: float,
    chi: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    psi_in = build_single_mode_phase_example_state(cutoff)
    phase_rotation = build_phase_rotation_matrix(cutoff, theta)
    kerr = build_kerr_matrix(cutoff, chi)

    examples = [
        ("Phase Rotation", wrap_phase_over_pi(np.diag(phase_rotation)), wrap_phase_over_pi(psi_in), wrap_phase_over_pi(np.diag(phase_rotation) * psi_in)),
        ("Kerr", wrap_phase_over_pi(np.diag(kerr)), wrap_phase_over_pi(psi_in), wrap_phase_over_pi(np.diag(kerr) * psi_in)),
    ]

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=4,
        nrows=2,
        panel_aspect=0.95,
    )
    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)

    image = None
    for row, (name, stored_phase, input_phase, output_phase) in enumerate(examples):
        image = render_scalar_heatmap(
            axes[row, 0],
            input_phase[np.newaxis, :],
            cmap=cmap,
            title=rf"{name}: input phase $\arg(\psi_{{in}})/\pi$",
            xlabel="Fock index $n$",
            ylabel="",
            vmin=0.0,
            vmax=2.0,
            aspect="auto",
        )
        axes[row, 0].set_yticks([])

        render_scalar_heatmap(
            axes[row, 1],
            stored_phase[np.newaxis, :],
            cmap=cmap,
            title=rf"{name}: stored diagonal $\phi[n]/\pi$",
            xlabel="Fock index $n$",
            ylabel="",
            vmin=0.0,
            vmax=2.0,
            aspect="auto",
        )
        axes[row, 1].set_yticks([])

        render_scalar_heatmap(
            axes[row, 2],
            output_phase[np.newaxis, :],
            cmap=cmap,
            title=rf"{name}: output phase $\arg(\psi_{{out}})/\pi$",
            xlabel="Fock index $n$",
            ylabel="",
            vmin=0.0,
            vmax=2.0,
            aspect="auto",
        )
        axes[row, 2].set_yticks([])
        axes[row, 2].text(
            0.5,
            -0.30,
            r"$\psi_{out}[n] = e^{i\phi[n]} \cdot \psi_{in}[n]$",
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=6.0,
            color=PALETTE_DARK,
        )

        render_heatmap(
            axes[row, 3],
            np.abs(psi_in)[np.newaxis, :],
            cmap=cmap,
            title=rf"{name}: preserved magnitude $|\psi_{{out}}[n]|$",
            xlabel="Fock index $n$",
            ylabel="",
            vmax=float(np.max(np.abs(psi_in))),
            aspect="auto",
        )
        axes[row, 3].set_yticks([])

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02, label=r"Wrapped phase / $\pi$")
    output_path = output_dir / "level0_single_mode_flow.svg"
    save_svg(fig, output_path)
    return output_path


def plot_level1_single_mode_heatmaps(
    output_dir: Path,
    cutoff: int,
    cmap: LinearSegmentedColormap,
) -> Path:
    creation = build_creation_matrix(cutoff)
    annihilation = build_annihilation_matrix(cutoff)
    creation_storage = level1_storage_vector("creation", cutoff)[np.newaxis, :]
    annihilation_storage = level1_storage_vector("annihilation", cutoff)[np.newaxis, :]

    vmax = float(
        max(
            np.max(np.abs(creation)),
            np.max(np.abs(annihilation)),
            np.max(creation_storage),
            np.max(annihilation_storage),
        )
    )

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=2,
        nrows=2,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.ravel()

    render_heatmap(
        axes_flat[0],
        creation,
        cmap=cmap,
        title=r"Creation operator $|a^\dagger|$",
        xlabel="Input Fock index $n$",
        ylabel="Output Fock index $m$",
        vmax=vmax,
    )
    render_heatmap(
        axes_flat[1],
        creation_storage,
        cmap=cmap,
        title=r"Creation storage vector $\sqrt{n}$",
        xlabel="Output Fock index $n$",
        ylabel="Stored coeff.",
        vmax=vmax,
        aspect="auto",
    )
    render_heatmap(
        axes_flat[2],
        annihilation,
        cmap=cmap,
        title=r"Annihilation operator $|a|$",
        xlabel="Input Fock index $n$",
        ylabel="Output Fock index $m$",
        vmax=vmax,
    )
    image = render_heatmap(
        axes_flat[3],
        annihilation_storage,
        cmap=cmap,
        title=r"Annihilation storage vector $\sqrt{n+1}$",
        xlabel="Output Fock index $n$",
        ylabel="Stored coeff.",
        vmax=vmax,
        aspect="auto",
    )

    fig.colorbar(image, ax=axes_flat.tolist(), fraction=0.022, pad=0.02, label="Magnitude")
    output_path = output_dir / "level1_single_mode_heatmaps.svg"
    save_svg(fig, output_path)
    return output_path


def plot_level1_single_mode_flow(
    output_dir: Path,
    cutoff: int,
    cmap: LinearSegmentedColormap,
) -> Path:
    psi_in = build_single_mode_example_state(cutoff)
    creation_coeff = level1_storage_vector("creation", cutoff)
    annihilation_coeff = level1_storage_vector("annihilation", cutoff)

    creation_shifted = np.zeros(cutoff, dtype=np.complex128)
    creation_shifted[1:] = psi_in[:-1]
    creation_out = creation_coeff * creation_shifted

    annihilation_shifted = np.zeros(cutoff, dtype=np.complex128)
    annihilation_shifted[:-1] = psi_in[1:]
    annihilation_out = annihilation_coeff * annihilation_shifted

    examples = [
        ("Creation", creation_shifted, creation_coeff, creation_out, r"$\psi_{out}[n] = \sqrt{n}\,\psi_{in}[n-1]$"),
        ("Annihilation", annihilation_shifted, annihilation_coeff, annihilation_out, r"$\psi_{out}[n] = \sqrt{n+1}\,\psi_{in}[n+1]$"),
    ]

    vmax = float(
        max(
            np.max(np.abs(psi_in)),
            np.max(creation_coeff),
            np.max(annihilation_coeff),
            np.max(np.abs(creation_out)),
            np.max(np.abs(annihilation_out)),
        )
    )

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=4,
        nrows=2,
        panel_aspect=0.95,
    )
    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)

    image = None
    for row, (name, shifted_input, coeffs, output_state, formula) in enumerate(examples):
        render_heatmap(
            axes[row, 0],
            np.abs(psi_in)[np.newaxis, :],
            cmap=cmap,
            title=rf"{name}: input magnitude $|\psi_{{in}}[n]|$",
            xlabel="Fock index $n$",
            ylabel="",
            vmax=vmax,
            aspect="auto",
        )
        axes[row, 0].set_yticks([])

        render_heatmap(
            axes[row, 1],
            np.abs(shifted_input)[np.newaxis, :],
            cmap=cmap,
            title=rf"{name}: shifted source slice",
            xlabel="Output index $n$",
            ylabel="",
            vmax=vmax,
            aspect="auto",
        )
        axes[row, 1].set_yticks([])

        render_heatmap(
            axes[row, 2],
            coeffs[np.newaxis, :],
            cmap=cmap,
            title=rf"{name}: stored coeff. vector",
            xlabel="Output index $n$",
            ylabel="",
            vmax=vmax,
            aspect="auto",
        )
        axes[row, 2].set_yticks([])
        axes[row, 2].text(
            0.5,
            -0.30,
            formula,
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=6.0,
            color=PALETTE_DARK,
        )

        image = render_heatmap(
            axes[row, 3],
            np.abs(output_state)[np.newaxis, :],
            cmap=cmap,
            title=rf"{name}: output magnitude $|\psi_{{out}}[n]|$",
            xlabel="Output index $n$",
            ylabel="",
            vmax=vmax,
            aspect="auto",
        )
        axes[row, 3].set_yticks([])

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02, label="Magnitude")
    output_path = output_dir / "level1_single_mode_flow.svg"
    save_svg(fig, output_path)
    return output_path


def plot_single_mode_level2(
    output_dir: Path,
    cutoff: int,
    alpha: complex,
    squeezing_r: float,
    squeezing_theta: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    displacement = build_displacement_matrix(cutoff, alpha)
    squeezing = build_squeezing_matrix(cutoff, squeezing_r, squeezing_theta)
    displacement_ell_values, _ = convert_to_ell_storage(displacement, ELL_THRESHOLD)
    squeezing_ell_values, _ = convert_to_ell_storage(squeezing, ELL_THRESHOLD)

    vmax = float(
        max(
            np.max(np.abs(displacement)),
            np.nanmax(np.abs(displacement_ell_values)),
            np.max(np.abs(squeezing)),
            np.nanmax(np.abs(squeezing_ell_values)),
        )
    )

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=2,
        nrows=2,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.ravel()

    render_heatmap(
        axes_flat[0],
        displacement,
        cmap=cmap,
        title=rf"Displacement $|D(\alpha)|$, $\alpha={alpha.real:.2f}{alpha.imag:+.2f}i$",
        xlabel="Input Fock index $n$",
        ylabel="Output Fock index $m$",
        vmax=vmax,
    )
    render_heatmap(
        axes_flat[1],
        displacement_ell_values,
        cmap=cmap,
        title=rf"Displacement ELL storage $|\mathrm{{ELL\_Val}}|$, $\tau=10^{{-12}}$",
        xlabel="ELL slot $k$",
        ylabel="Row $m$",
        vmax=vmax,
        mask_invalid=True,
    )
    render_heatmap(
        axes_flat[2],
        squeezing,
        cmap=cmap,
        title=rf"Squeezing $|S(r,\theta)|$, $r={squeezing_r:.2f}$",
        xlabel="Input Fock index $n$",
        ylabel="Output Fock index $m$",
        vmax=vmax,
    )
    image = render_heatmap(
        axes_flat[3],
        squeezing_ell_values,
        cmap=cmap,
        title=rf"Squeezing ELL storage $|\mathrm{{ELL\_Val}}|$, $\tau=10^{{-12}}$",
        xlabel="ELL slot $k$",
        ylabel="Row $m$",
        vmax=vmax,
        mask_invalid=True,
    )

    fig.colorbar(image, ax=axes_flat.tolist(), fraction=0.022, pad=0.02, label="Magnitude")
    output_path = output_dir / "level2_single_mode_heatmaps.svg"
    save_svg(fig, output_path)
    return output_path


def plot_single_mode_level2_flow(
    output_dir: Path,
    cutoff: int,
    squeezing_r: float,
    squeezing_theta: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    operator_dense = build_squeezing_matrix(cutoff, squeezing_r, squeezing_theta)
    sparse_operator = np.where(np.abs(operator_dense) > ELL_THRESHOLD, operator_dense, np.nan + 0.0j)
    psi_in = build_single_mode_example_state(cutoff)
    psi_out = operator_dense @ psi_in
    target_row = min(4, cutoff - 1)
    selected_cols = np.where(np.abs(operator_dense[target_row]) > ELL_THRESHOLD)[0]
    row_values = operator_dense[target_row, selected_cols]
    gathered_input = psi_in[selected_cols]
    contributions = row_values * gathered_input
    flow_panel = np.full((3, cutoff), np.nan + 0.0j, dtype=np.complex128)
    flow_panel[0, selected_cols] = row_values
    flow_panel[1, selected_cols] = gathered_input
    flow_panel[2, selected_cols] = contributions

    vmax_matrix = float(np.nanmax(np.abs(sparse_operator)))
    vmax_state = float(
        max(
            np.max(np.abs(psi_in)),
            np.max(np.abs(psi_out)),
            np.nanmax(np.abs(flow_panel)),
        )
    )

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=4,
        nrows=1,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)

    render_heatmap(
        axes[0],
        psi_in[np.newaxis, :],
        cmap=cmap,
        title=r"Single-mode input vector $|\psi_{in}[n]|$",
        xlabel="Input Fock index $n$",
        ylabel="",
        vmax=vmax_state,
    )
    for col in selected_cols:
        axes[0].add_patch(Rectangle((col - 0.5, -0.5), 1.0, 1.0, fill=False, edgecolor=PALETTE_DARK, linewidth=0.8))
    axes[0].text(
        0.5,
        -0.18,
        rf"Highlighted entries are gathered to compute row $n={target_row}$.",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )
    axes[0].set_yticks([])

    render_heatmap(
        axes[1],
        sparse_operator,
        cmap=cmap,
        title=r"Squeezing ELL support, checkerboard parity pattern",
        xlabel="Input index $n$",
        ylabel="Output index $m$",
        vmax=vmax_matrix,
        mask_invalid=True,
    )
    axes[1].add_patch(Rectangle((-0.5, target_row - 0.5), cutoff, 1.0, fill=False, edgecolor=PALETTE_DARK, linewidth=0.9))
    for col in selected_cols:
        axes[1].add_patch(Rectangle((col - 0.5, target_row - 0.5), 1.0, 1.0, fill=False, edgecolor=PALETTE_DARK, linewidth=1.0))

    render_heatmap(
        axes[2],
        flow_panel,
        cmap=cmap,
        title=rf"ELL gather for output row $n={target_row}$",
        xlabel="Selected input index from ELL_Col",
        ylabel="",
        vmax=vmax_state,
        mask_invalid=True,
    )
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels([r"$|\mathrm{ELL\_Val}|$", r"$|\psi_{in}|$", r"$|\mathrm{ELL\_Val}\cdot\psi_{in}|$"])
    axes[2].text(
        0.5,
        -0.18,
        r"$\psi_{out}[n] = \sum_k \mathrm{ELL\_Val}[n][k]\cdot\psi_{in}[\mathrm{ELL\_Col}[n][k]]$",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    image = render_heatmap(
        axes[3],
        psi_out[np.newaxis, :],
        cmap=cmap,
        title=r"Single-mode output vector $|\psi_{out}[m]|$",
        xlabel="Output Fock index $m$",
        ylabel="",
        vmax=vmax_state,
    )
    axes[3].add_patch(Rectangle((target_row - 0.5, -0.5), 1.0, 1.0, fill=False, edgecolor=PALETTE_DARK, linewidth=0.9))
    axes[3].set_yticks([])
    axes[3].text(
        0.5,
        -0.18,
        "Parity conservation gives the stride-2 checkerboard access pattern.",
        transform=axes[3].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    fig.colorbar(image, ax=axes, fraction=0.020, pad=0.02, label="Magnitude")
    output_path = output_dir / "level2_single_mode_flow.svg"
    save_svg(fig, output_path)
    return output_path


def plot_two_qumode_single_mode_parallel_flow(
    output_dir: Path,
    cutoff: int,
    alpha: complex,
    cmap: LinearSegmentedColormap,
) -> Path:
    state_in = build_two_qumode_parallel_example_state(cutoff)
    flat_in = state_in.reshape(-1)
    local_gate = build_displacement_matrix(cutoff, alpha)
    state_out = apply_single_mode_matrix_on_last_mode(state_in, local_gate)
    flat_out = state_out.reshape(-1)

    vmax = float(
        max(
            np.max(np.abs(state_in)),
            np.max(np.abs(state_out)),
            np.max(np.abs(flat_in)),
            np.max(np.abs(flat_out)),
            np.max(np.abs(local_gate)),
        )
    )

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=3,
        nrows=2,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    flat_image = render_heatmap(
        axes[0, 0],
        flat_in[:, np.newaxis],
        cmap=cmap,
        title=r"Flattened input storage $|\psi_{in}[idx]|$, $idx=mD+n$",
        xlabel="Storage lane",
        ylabel=r"Flat index $idx$",
        vmax=vmax,
        aspect="equal",
    )
    axes[0, 0].set_xticks([])
    chunk_centers = [row * cutoff + 0.5 * (cutoff - 1) for row in range(cutoff)]
    axes[0, 0].set_yticks(chunk_centers)
    axes[0, 0].set_yticklabels([rf"$m={row}$" for row in range(cutoff)])
    for row in range(1, cutoff):
        axes[0, 0].axhline(row * cutoff - 0.5, color="#7A5834", linewidth=0.45, alpha=0.55)
    draw_rect_lattice(axes[0, 0], cutoff * cutoff, 1)
    axes[0, 0].text(
        0.5,
        -0.28,
        "Each contiguous vertical chunk of length D is one fixed-spectator slice.",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    render_heatmap(
        axes[0, 1],
        state_in,
        cmap=cmap,
        title=r"Same data in 2D view $|\psi_{in}[m,n]|$",
        xlabel="Target-mode photons $n$",
        ylabel="Spectator photons $m$",
        vmax=vmax,
        aspect="equal",
    )
    draw_square_lattice(axes[0, 1], cutoff)

    render_heatmap(
        axes[0, 2],
        state_in,
        cmap=cmap,
        title=r"Parallel row slices (same data, compute view)",
        xlabel="Local target index $n$",
        ylabel="Slice id = spectator $m$",
        vmax=vmax,
    )
    draw_square_lattice(axes[0, 2], cutoff)
    axes[0, 2].text(
        0.5,
        -0.28,
        "The 2D view is only for understanding the parallel row slices.",
        transform=axes[0, 2].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    render_heatmap(
        axes[1, 0],
        local_gate,
        cmap=cmap,
        title=rf"Example single-mode gate $|D(\alpha)|$, $\alpha={alpha.real:.2f}{alpha.imag:+.2f}i$",
        xlabel="Input local index",
        ylabel="Output local index",
        vmax=vmax,
    )
    axes[1, 0].text(
        0.5,
        -0.24,
        r"$\psi_{out}[m,:] = U \cdot \psi_{in}[m,:]$, for all fixed $m$ in parallel",
        transform=axes[1, 0].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    render_heatmap(
        axes[1, 1],
        state_out,
        cmap=cmap,
        title=r"Parallel row outputs before write-back",
        xlabel="Local target index $n$",
        ylabel="Slice id = spectator $m$",
        vmax=vmax,
    )
    draw_square_lattice(axes[1, 1], cutoff)

    render_heatmap(
        axes[1, 2],
        flat_out[:, np.newaxis],
        cmap=cmap,
        title=r"Flattened output storage $|\psi_{out}[idx]|$",
        xlabel="Storage lane",
        ylabel=r"Flat index $idx$",
        vmax=vmax,
        aspect="equal",
    )
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks(chunk_centers)
    axes[1, 2].set_yticklabels([rf"$m={row}$" for row in range(cutoff)])
    for row in range(1, cutoff):
        axes[1, 2].axhline(row * cutoff - 0.5, color="#7A5834", linewidth=0.45, alpha=0.55)
    draw_rect_lattice(axes[1, 2], cutoff * cutoff, 1)
    axes[1, 2].text(
        0.5,
        -0.24,
        "After parallel slice updates, the runtime writes results back to the same flat layout.",
        transform=axes[1, 2].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    fig.colorbar(flat_image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02, label="Magnitude")
    output_path = output_dir / "two_qumode_single_mode_parallel_flow.svg"
    save_svg(fig, output_path)
    return output_path


def plot_two_qumode_single_mode_stride_parallel_flow(
    output_dir: Path,
    cutoff: int,
    alpha: complex,
    cmap: LinearSegmentedColormap,
) -> Path:
    state_in = build_two_qumode_parallel_example_state(cutoff)
    flat_in = state_in.reshape(-1)
    local_gate = build_displacement_matrix(cutoff, alpha)
    state_out = apply_single_mode_matrix_on_first_mode(state_in, local_gate)
    flat_out = state_out.reshape(-1)

    vmax = float(
        max(
            np.max(np.abs(state_in)),
            np.max(np.abs(state_out)),
            np.max(np.abs(flat_in)),
            np.max(np.abs(local_gate)),
        )
    )

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=3,
        nrows=2,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    flat_image = render_heatmap(
        axes[0, 0],
        flat_in[:, np.newaxis],
        cmap=cmap,
        title=r"Flattened input storage $|\psi_{in}[idx]|$, $idx=mD+n$",
        xlabel="Storage lane",
        ylabel=r"Flat index $idx$",
        vmax=vmax,
        aspect="equal",
    )
    axes[0, 0].set_xticks([])
    chunk_centers = [row * cutoff + 0.5 * (cutoff - 1) for row in range(cutoff)]
    axes[0, 0].set_yticks(chunk_centers)
    axes[0, 0].set_yticklabels([rf"$m={row}$" for row in range(cutoff)])
    for row in range(1, cutoff):
        axes[0, 0].axhline(row * cutoff - 0.5, color="#7A5834", linewidth=0.45, alpha=0.55)
    draw_rect_lattice(axes[0, 0], cutoff * cutoff, 1)

    render_heatmap(
        axes[0, 1],
        state_in,
        cmap=cmap,
        title=r"Same data in 2D view $|\psi_{in}[m,n]|$",
        xlabel="Spectator photons $n$",
        ylabel="Target-mode photons $m$",
        vmax=vmax,
        aspect="equal",
    )
    draw_square_lattice(axes[0, 1], cutoff)

    highlight_ns = []
    for candidate in [0, min(2, cutoff - 1), cutoff - 1]:
        if candidate not in highlight_ns:
            highlight_ns.append(candidate)
    styles = ["-", "--", "-."]
    for style_idx, spectator_n in enumerate(highlight_ns):
        for target_m in range(cutoff):
            flat_idx = target_m * cutoff + spectator_n
            axes[0, 1].add_patch(
                Rectangle(
                    (spectator_n - 0.5, target_m - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=PALETTE_DARK,
                    linewidth=0.85,
                    linestyle=styles[style_idx % len(styles)],
                )
            )
            axes[0, 0].add_patch(
                Rectangle(
                    (-0.5, flat_idx - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=PALETTE_DARK,
                    linewidth=0.85,
                    linestyle=styles[style_idx % len(styles)],
                )
            )
    axes[0, 0].text(
        0.5,
        -0.28,
        r"For fixed spectator $n$, source indices are $n, D+n, 2D+n, \ldots$ with stride $D$.",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    render_heatmap(
        axes[0, 2],
        state_in.T,
        cmap=cmap,
        title=r"Parallel column slices (compute view)",
        xlabel="Local target index $m$",
        ylabel="Slice id = spectator $n$",
        vmax=vmax,
    )
    draw_square_lattice(axes[0, 2], cutoff)
    axes[0, 2].text(
        0.5,
        -0.28,
        "The 2D view is only for understanding the gathered stride-D column slices.",
        transform=axes[0, 2].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    render_heatmap(
        axes[1, 0],
        local_gate,
        cmap=cmap,
        title=rf"Example single-mode gate $|D(\alpha)|$, $\alpha={alpha.real:.2f}{alpha.imag:+.2f}i$",
        xlabel="Input local index",
        ylabel="Output local index",
        vmax=vmax,
    )
    axes[1, 0].text(
        0.5,
        -0.24,
        r"$\psi_{out}[:,n] = U \cdot \psi_{in}[:,n]$, for all fixed $n$ in parallel",
        transform=axes[1, 0].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    render_heatmap(
        axes[1, 1],
        state_out.T,
        cmap=cmap,
        title=r"Parallel gathered outputs before write-back",
        xlabel="Local target index $m$",
        ylabel="Slice id = spectator $n$",
        vmax=vmax,
    )
    draw_square_lattice(axes[1, 1], cutoff)

    render_heatmap(
        axes[1, 2],
        flat_out[:, np.newaxis],
        cmap=cmap,
        title=r"Flattened output storage $|\psi_{out}[idx]|$",
        xlabel="Storage lane",
        ylabel=r"Flat index $idx$",
        vmax=vmax,
        aspect="equal",
    )
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks(chunk_centers)
    axes[1, 2].set_yticklabels([rf"$m={row}$" for row in range(cutoff)])
    for row in range(1, cutoff):
        axes[1, 2].axhline(row * cutoff - 0.5, color="#7A5834", linewidth=0.45, alpha=0.55)
    draw_rect_lattice(axes[1, 2], cutoff * cutoff, 1)
    axes[1, 2].text(
        0.5,
        -0.24,
        "After gathered slice updates, the runtime writes results back to the same flat layout.",
        transform=axes[1, 2].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    fig.colorbar(flat_image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02, label="Magnitude")
    output_path = output_dir / "two_qumode_single_mode_stride_parallel_flow.svg"
    save_svg(fig, output_path)
    return output_path


def plot_single_mode_ell_overview(
    output_dir: Path,
    cutoff: int,
    squeezing_r: float,
    squeezing_theta: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    operator_dense = build_squeezing_matrix(cutoff, squeezing_r, squeezing_theta)
    sparse_operator = np.where(np.abs(operator_dense) > ELL_THRESHOLD, operator_dense, np.nan + 0.0j)
    ell_values, ell_cols = convert_to_ell_storage(operator_dense, ELL_THRESHOLD)

    vmax_mag = float(max(np.nanmax(np.abs(sparse_operator)), np.nanmax(np.abs(ell_values))))
    ell_col_masked = np.ma.masked_where(ell_cols < 0, ell_cols.astype(np.float64))

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=3,
        nrows=1,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    render_heatmap(
        axes[0],
        sparse_operator,
        cmap=cmap,
        title=r"Thresholded squeezing support in Fock order",
        xlabel="Input Fock index $n$",
        ylabel="Output Fock index $m$",
        vmax=vmax_mag,
        mask_invalid=True,
    )

    index_image = axes[1].imshow(
        ell_col_masked,
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=float(cutoff - 1),
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title(r"Packed ELL\_Col index map")
    axes[1].set_xlabel(r"ELL slot $k$")
    axes[1].set_ylabel(r"Row $m$")
    axes[1].grid(False)

    value_image = render_heatmap(
        axes[2],
        ell_values,
        cmap=cmap,
        title=r"Packed ELL\_Val magnitude map",
        xlabel=r"ELL slot $k$",
        ylabel=r"Row $m$",
        vmax=vmax_mag,
        mask_invalid=True,
    )

    axes[1].text(
        0.5,
        -0.18,
        r"Each row stores only surviving columns, packed left to right into contiguous ELL slots.",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )
    axes[2].text(
        0.5,
        -0.18,
        r"Parity conservation makes the surviving entries follow the stride-2 checkerboard pattern.",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    cbar_mag = fig.colorbar(value_image, ax=[axes[0], axes[2]], fraction=0.024, pad=0.02)
    cbar_mag.set_label("Magnitude")
    cbar_idx = fig.colorbar(index_image, ax=[axes[1]], fraction=0.046, pad=0.02)
    cbar_idx.set_label("Stored input index")

    output_path = output_dir / "level2_ell_overview.svg"
    save_svg(fig, output_path)
    return output_path


def plot_bs_dense(
    output_dir: Path,
    cutoff: int,
    theta: float,
    phi: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    dense = build_bs_dense_matrix(cutoff, theta, phi)
    separators = [i * cutoff for i in range(1, cutoff)]
    vmax = float(np.max(np.abs(dense)))

    figsize = apply_paper_style(
        width_pt=SINGLE_COLUMN_PT,
        ncols=1,
        nrows=1,
        panel_aspect=1.0,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    image = render_heatmap(
        ax,
        dense,
        cmap=cmap,
        title=rf"Beam splitter dense matrix $|BS(\theta,\phi)|$, $D={cutoff}$",
        xlabel=r"Input basis index $(p,q) \mapsto pD+q$",
        ylabel=r"Output basis index $(m,n) \mapsto mD+n$",
        vmax=vmax,
        add_separators=separators,
    )
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, label="Magnitude")

    output_path = output_dir / "beam_splitter_dense_heatmap.svg"
    save_svg(fig, output_path)
    return output_path


def plot_bs_sector_overview(
    output_dir: Path,
    cutoff: int,
    theta: float,
    phi: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    dense = build_bs_dense_matrix(cutoff, theta, phi)
    sector_view, separators = reorder_bs_matrix_by_sector(dense, cutoff)
    vmax = float(np.max(np.abs(sector_view)))

    ordered_indices, _ = sector_order_indices(cutoff)
    sector_sizes = [bs_sector_size(cutoff, L) for L in range(2 * cutoff - 1)]

    centers: list[float] = []
    cursor = 0
    for size in sector_sizes:
        centers.append(cursor + 0.5 * (size - 1))
        cursor += size

    figsize = apply_paper_style(
        width_pt=SINGLE_COLUMN_PT,
        ncols=1,
        nrows=1,
        panel_aspect=1.0,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    image = render_heatmap(
        ax,
        sector_view,
        cmap=cmap,
        title=rf"Beam splitter overview grouped by total photon number $L=m+n$",
        xlabel="Input basis ordered by sector $L$",
        ylabel="Output basis ordered by sector $L$",
        vmax=vmax,
        add_separators=separators,
    )

    labels = [str(L) for L in range(len(centers))]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels)
    ax.text(
        0.5,
        -0.16,
        r"Each diagonal block is one cached submatrix $U_{BS}^{(L)}$; off-block entries are zero by photon-number conservation.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=PALETTE_DARK,
    )

    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, label="Magnitude")
    output_path = output_dir / "beam_splitter_sector_overview.svg"
    save_svg(fig, output_path)
    return output_path


def plot_bs_flow(
    output_dir: Path,
    cutoff: int,
    theta: float,
    phi: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    tensor = build_bs_tensor_recursive(cutoff, theta, phi)
    dense = tensor.reshape(cutoff * cutoff, cutoff * cutoff)
    sector_view, separators = reorder_bs_matrix_by_sector(dense, cutoff)
    subspaces = build_bs_subspace_matrices(cutoff, theta, phi)
    sector_sizes: list[int] = []
    sector_starts: list[int] = []
    sector_centers: list[float] = []
    cursor = 0
    for L in range(2 * cutoff - 1):
        size = bs_sector_size(cutoff, L)
        sector_sizes.append(size)
        sector_starts.append(cursor)
        sector_centers.append(cursor + 0.5 * (size - 1))
        cursor += size

    example_sectors: list[int] = []
    for candidate in [min(2, cutoff - 1), min(4, cutoff - 1), cutoff - 1]:
        if candidate >= 0 and candidate not in example_sectors:
            example_sectors.append(candidate)
    if not example_sectors:
        example_sectors = [0]

    global_input = build_bs_multi_sector_example_state(cutoff, example_sectors)
    global_output = apply_two_mode_tensor_to_state(tensor, global_input)

    example_data: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    vmax = float(max(np.max(np.abs(dense)), np.max(np.abs(sector_view))))
    for L in example_sectors:
        local_input = extract_sector_vector(global_input, L)
        local_output = extract_sector_vector(global_output, L)
        submatrix = subspaces[L]
        vmax = float(
            max(
                vmax,
                np.max(np.abs(global_input)),
                np.max(np.abs(global_output)),
                np.max(np.abs(local_input)),
                np.max(np.abs(local_output)),
                np.max(np.abs(submatrix)),
            )
        )
        example_data.append((L, local_input, submatrix, local_output))

    fig = plt.figure(figsize=(8.8, 9.4), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.935, top=0.955, bottom=0.070)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 2.85], hspace=0.18)
    top = outer[0].subgridspec(1, 2, wspace=0.18)
    bottom = outer[1].subgridspec(
        len(example_data),
        5,
        width_ratios=[1.35, 0.92, 1.00, 0.92, 1.35],
        hspace=0.24,
        wspace=0.16,
    )

    ax_dense = fig.add_subplot(top[0, 0])
    ax_sector = fig.add_subplot(top[0, 1])
    all_axes = [ax_dense, ax_sector]

    separators_dense = [i * cutoff for i in range(1, cutoff)]
    image = render_heatmap(
        ax_dense,
        dense,
        cmap=cmap,
        title=rf"Lexicographic dense $|BS(\theta,\phi)|$, $D={cutoff}$",
        xlabel=r"Input basis index $(p,q)\mapsto pD+q$",
        ylabel=r"Output basis index $(m,n)\mapsto mD+n$",
        vmax=vmax,
        add_separators=separators_dense,
    )
    ax_dense.set_box_aspect(1.0)

    render_heatmap(
        ax_sector,
        sector_view,
        cmap=cmap,
        title=r"Same matrix reordered by total photon number $L=m+n$",
        xlabel="Input basis ordered by sector $L$",
        ylabel="Output basis ordered by sector $L$",
        vmax=vmax,
        add_separators=separators,
    )
    ax_sector.set_box_aspect(1.0)
    ax_sector.set_xticks(sector_centers)
    ax_sector.set_xticklabels([str(L) for L in range(len(sector_centers))])
    ax_sector.set_yticks(sector_centers)
    ax_sector.set_yticklabels([str(L) for L in range(len(sector_centers))])
    line_styles = ["-", "--", "-."]
    for idx, L in enumerate(example_sectors):
        start = sector_starts[L]
        size = sector_sizes[L]
        ax_sector.add_patch(
            Rectangle(
                (start - 0.5, start - 0.5),
                size,
                size,
                fill=False,
                edgecolor=PALETTE_DARK,
                linewidth=1.0,
                linestyle=line_styles[idx % len(line_styles)],
            )
        )

    ax_global_in = fig.add_subplot(bottom[:, 0])
    render_heatmap(
        ax_global_in,
        global_input,
        cmap=cmap,
        title=r"Initial 2D Fock state $|\psi_{in}[m,n]|$",
        xlabel="Mode-2 photons $n$",
        ylabel="Mode-1 photons $m$",
        vmax=vmax,
    )
    draw_square_lattice(ax_global_in, cutoff)
    all_axes.append(ax_global_in)

    ax_global_out = fig.add_subplot(bottom[:, 4])
    render_heatmap(
        ax_global_out,
        global_output,
        cmap=cmap,
        title=r"Final 2D Fock state $|\psi_{out}[m,n]|$",
        xlabel="Mode-2 photons $n$",
        ylabel="Mode-1 photons $m$",
        vmax=vmax,
    )
    draw_square_lattice(ax_global_out, cutoff)
    all_axes.append(ax_global_out)

    for row_idx, (L, local_input, submatrix, local_output) in enumerate(example_data, start=1):
        row = row_idx - 1
        ax_in = fig.add_subplot(bottom[row, 1])
        ax_op = fig.add_subplot(bottom[row, 2])
        ax_out = fig.add_subplot(bottom[row, 3])
        all_axes.extend([ax_in, ax_op, ax_out])

        render_heatmap(
            ax_in,
            local_input[np.newaxis, :],
            cmap=cmap,
            title=rf"Example {row_idx}: $\psi_{{in}}^{{({L})}}$",
            xlabel=r"col = mode-2 photons",
            ylabel="",
            vmax=vmax,
        )
        ax_in.set_yticks([])

        render_heatmap(
            ax_op,
            submatrix,
            cmap=cmap,
            title=rf"Sector block $U_{{BS}}^{{({L})}}$",
            xlabel="col",
            ylabel="row",
            vmax=vmax,
        )
        if row_idx == 1:
            ax_op.text(
                0.5,
                -0.18,
                rf"$\psi_{{out}}^{{({L})}}[row] = \sum_{{col=0}}^{{{submatrix.shape[1] - 1}}} U^{{({L})}}_{{row,col}} \cdot \psi_{{in}}^{{({L})}}[col]$",
                transform=ax_op.transAxes,
                ha="center",
                va="top",
                fontsize=6.0,
                color=PALETTE_DARK,
            )

        render_heatmap(
            ax_out,
            local_output[np.newaxis, :],
            cmap=cmap,
            title=rf"Example {row_idx}: $\psi_{{out}}^{{({L})}}$",
            xlabel=r"row = mode-2 photons",
            ylabel="",
            vmax=vmax,
        )
        ax_out.set_yticks([])

    line_styles = ["-", "--", "-."]
    for idx, L in enumerate(example_sectors):
        for mode2 in range(L + 1):
            mode1 = L - mode2
            if 0 <= mode1 < cutoff and 0 <= mode2 < cutoff:
                style = line_styles[idx % len(line_styles)]
                for ax in [ax_global_in, ax_global_out]:
                    ax.add_patch(
                        Rectangle(
                            (mode2 - 0.5, mode1 - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor=PALETTE_DARK,
                            linewidth=0.9,
                            linestyle=style,
                        )
                    )

    fig.colorbar(image, ax=all_axes, fraction=0.016, pad=0.02, label="Magnitude")
    output_path = output_dir / "beam_splitter_flow.svg"
    save_svg(fig, output_path)
    return output_path


def plot_bs_subspaces(
    output_dir: Path,
    cutoff: int,
    theta: float,
    phi: float,
    cmap: LinearSegmentedColormap,
) -> Path:
    subspaces = build_bs_subspace_matrices(cutoff, theta, phi)
    vmax = float(max(np.max(np.abs(sub)) for sub in subspaces))
    panel_count = len(subspaces)
    ncols = math.ceil(math.sqrt(panel_count))
    nrows = math.ceil(panel_count / ncols)

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=ncols,
        nrows=nrows,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    image = None

    for k, sub in enumerate(subspaces):
        row = k // ncols
        col = k % ncols
        ax = axes_arr[row, col]
        image = render_heatmap(
            ax,
            sub,
            cmap=cmap,
            title=rf"$k={k}$, offset $O_k={subspace_offset(k)}$",
            xlabel=r"Input mode-2 photons $j$",
            ylabel=r"Output mode-2 photons $i$",
            vmax=vmax,
        )

    for idx in range(panel_count, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes_arr[row, col].axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes_arr.ravel().tolist(), fraction=0.018, pad=0.02, label="Magnitude")

    fig.suptitle(
        rf"Beam splitter photon-number sector blocks $U^{{(k)}}$ for $BS(\theta={theta:.2f},\phi={phi:.2f})$",
        y=1.02,
    )

    output_path = output_dir / "beam_splitter_subspace_heatmaps.svg"
    save_svg(fig, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-cutoff", type=int, default=8, help="Cutoff for Level 2 single-mode examples.")
    parser.add_argument("--two-mode-cutoff", type=int, default=6, help="Cutoff for beam-splitter examples.")
    parser.add_argument("--phase-theta", type=float, default=0.55, help="Phase rotation angle theta for Level 0 examples.")
    parser.add_argument("--kerr-chi", type=float, default=0.22, help="Kerr parameter chi for Level 0 examples.")
    parser.add_argument("--disp-real", type=float, default=1.0, help="Real part of displacement alpha.")
    parser.add_argument("--disp-imag", type=float, default=0.35, help="Imaginary part of displacement alpha.")
    parser.add_argument("--squeezing-r", type=float, default=0.7, help="Squeezing amplitude r.")
    parser.add_argument("--squeezing-theta", type=float, default=0.0, help="Squeezing angle theta.")
    parser.add_argument("--bs-theta", type=float, default=0.18, help="Beam splitter theta.")
    parser.add_argument("--bs-phi", type=float, default=0.30, help="Beam splitter phi.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for SVG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmap = build_colormap()
    alpha = complex(args.disp_real, args.disp_imag)

    saved = [
        plot_level0_single_mode_heatmaps(
            output_dir=args.output_dir,
            cutoff=args.single_cutoff,
            theta=args.phase_theta,
            chi=args.kerr_chi,
            cmap=cmap,
        ),
        plot_level0_single_mode_flow(
            output_dir=args.output_dir,
            cutoff=args.single_cutoff,
            theta=args.phase_theta,
            chi=args.kerr_chi,
            cmap=cmap,
        ),
        plot_level1_single_mode_heatmaps(
            output_dir=args.output_dir,
            cutoff=args.single_cutoff,
            cmap=cmap,
        ),
        plot_level1_single_mode_flow(
            output_dir=args.output_dir,
            cutoff=args.single_cutoff,
            cmap=cmap,
        ),
        plot_single_mode_level2(
            output_dir=args.output_dir,
            cutoff=args.single_cutoff,
            alpha=alpha,
            squeezing_r=args.squeezing_r,
            squeezing_theta=args.squeezing_theta,
            cmap=cmap,
        ),
        plot_single_mode_level2_flow(
            output_dir=args.output_dir,
            cutoff=args.single_cutoff,
            squeezing_r=args.squeezing_r,
            squeezing_theta=args.squeezing_theta,
            cmap=cmap,
        ),
        plot_two_qumode_single_mode_parallel_flow(
            output_dir=args.output_dir,
            cutoff=args.two_mode_cutoff,
            alpha=alpha,
            cmap=cmap,
        ),
        plot_two_qumode_single_mode_stride_parallel_flow(
            output_dir=args.output_dir,
            cutoff=args.two_mode_cutoff,
            alpha=alpha,
            cmap=cmap,
        ),
        plot_single_mode_ell_overview(
            output_dir=args.output_dir,
            cutoff=args.single_cutoff,
            squeezing_r=args.squeezing_r,
            squeezing_theta=args.squeezing_theta,
            cmap=cmap,
        ),
        plot_bs_dense(
            output_dir=args.output_dir,
            cutoff=args.two_mode_cutoff,
            theta=args.bs_theta,
            phi=args.bs_phi,
            cmap=cmap,
        ),
        plot_bs_sector_overview(
            output_dir=args.output_dir,
            cutoff=args.two_mode_cutoff,
            theta=args.bs_theta,
            phi=args.bs_phi,
            cmap=cmap,
        ),
        plot_bs_flow(
            output_dir=args.output_dir,
            cutoff=args.two_mode_cutoff,
            theta=args.bs_theta,
            phi=args.bs_phi,
            cmap=cmap,
        ),
        plot_bs_subspaces(
            output_dir=args.output_dir,
            cutoff=args.two_mode_cutoff,
            theta=args.bs_theta,
            phi=args.bs_phi,
            cmap=cmap,
        ),
    ]

    for path in saved:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
