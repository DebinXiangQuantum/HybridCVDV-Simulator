#!/usr/bin/env python3
"""Generate SVG examples for symbolic Gaussian execution paths."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Ellipse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.configs.paper_style import DOUBLE_COLUMN_PT, apply_paper_style


PALETTE_LIGHT = "#FFDDAB"
PALETTE_MID = "#D18656"
PALETTE_DARK = "#945034"
GRID_COLOR = "#7A5834"
AXIS_MIN = -2.0
AXIS_MAX = 2.0
PROJECTION_DARK_COLORS = ["#5F8B4C", "#FF9A9A", "#945034"]
PROJECTION_LIGHT_COLORS = ["#D9E7D0", "#FFE1E1", "#E8D5CC"]


def build_colormap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "symbolic_gaussian",
        ["#FFF5E7", PALETTE_LIGHT, PALETTE_MID, PALETTE_DARK],
    )
    cmap.set_bad("#FFF8EE")
    return cmap


def rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float64,
    )


def gaussian_density_grid(
    mean: np.ndarray,
    cov: np.ndarray,
    q_grid: np.ndarray,
    p_grid: np.ndarray,
) -> np.ndarray:
    pos = np.stack([q_grid - mean[0], p_grid - mean[1]], axis=-1)
    inv_cov = np.linalg.inv(cov)
    exponent = np.einsum("...i,ij,...j->...", pos, inv_cov, pos)
    norm = 1.0 / (2.0 * math.pi * math.sqrt(float(np.linalg.det(cov))))
    return norm * np.exp(-0.5 * exponent)


def covariance_geometry(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def draw_covariance_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    edgecolor: str,
    linewidth: float = 1.2,
    linestyle: str = "-",
    alpha: float = 0.95,
) -> None:
    eigvals, eigvecs = covariance_geometry(cov)
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2.0 * math.sqrt(max(eigvals[0], 1e-12))
    height = 2.0 * math.sqrt(max(eigvals[1], 1e-12))

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(ellipse)
    ax.scatter([mean[0]], [mean[1]], s=11, c=edgecolor, zorder=5, alpha=alpha)


def setup_phase_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.35, alpha=0.28, color=GRID_COLOR, linestyle="--")


def gaussian_gate_example_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_in = np.array([-0.7, 0.35], dtype=np.float64)
    cov_in = np.array([[0.34, 0.08], [0.08, 0.19]], dtype=np.float64)

    r = 0.65
    phi = 0.52
    displacement = np.array([1.15, -0.75], dtype=np.float64)
    gate_linear = rotation(phi) @ np.diag([math.exp(-r), math.exp(r)]) @ rotation(-phi)

    mean_out = gate_linear @ mean_in + displacement
    cov_out = gate_linear @ cov_in @ gate_linear.T
    return mean_in, cov_in, mean_out, cov_out


def nongaussian_mixture_example_data() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base_components = [
        {
            "weight": 0.62,
            "mean": np.array([-0.95, 0.55], dtype=np.float64),
            "cov": np.array([[0.22, 0.04], [0.04, 0.13]], dtype=np.float64),
            "color": PALETTE_DARK,
            "linestyle": "-",
        },
        {
            "weight": 0.38,
            "mean": np.array([0.90, -0.45], dtype=np.float64),
            "cov": np.array([[0.18, -0.03], [-0.03, 0.25]], dtype=np.float64),
            "color": PALETTE_MID,
            "linestyle": "--",
        },
    ]

    coeffs = np.array([0.50, 0.32, 0.18], dtype=np.float64)
    phis = np.array([-0.34, 0.06, 0.41], dtype=np.float64)

    child_components: list[dict[str, object]] = []
    for component in base_components:
        for coeff, phi in zip(coeffs, phis):
            rot = rotation(float(phi))
            child_components.append(
                {
                    "weight": float(component["weight"]) * float(coeff),
                    "mean": rot @ np.asarray(component["mean"]),
                    "cov": rot @ np.asarray(component["cov"]) @ rot.T,
                    "color": component["color"],
                    "linestyle": component["linestyle"],
                }
            )
    return base_components, child_components


def gaussian_mixture_projection_example_data() -> list[dict[str, object]]:
    return [
        {
            "weight": 0.42,
            "mean": np.array([-1.05, 0.35], dtype=np.float64),
            "cov": np.array([[0.24, 0.05], [0.05, 0.14]], dtype=np.float64),
            "color": PROJECTION_DARK_COLORS[0],
            "light_color": PROJECTION_LIGHT_COLORS[0],
            "linestyle": "-",
            "weight_phase": -0.28,
        },
        {
            "weight": 0.33,
            "mean": np.array([0.05, 0.85], dtype=np.float64),
            "cov": np.array([[0.16, -0.02], [-0.02, 0.31]], dtype=np.float64),
            "color": PROJECTION_DARK_COLORS[1],
            "light_color": PROJECTION_LIGHT_COLORS[1],
            "linestyle": "--",
            "weight_phase": 0.36,
        },
        {
            "weight": 0.25,
            "mean": np.array([0.95, -0.55], dtype=np.float64),
            "cov": np.array([[0.20, 0.03], [0.03, 0.18]], dtype=np.float64),
            "color": PROJECTION_DARK_COLORS[2],
            "light_color": PROJECTION_LIGHT_COLORS[2],
            "linestyle": "-.",
            "weight_phase": 0.74,
        },
    ]


def coherent_fock_amplitudes(alpha: complex, cutoff: int) -> np.ndarray:
    amps = np.zeros(cutoff, dtype=np.complex128)
    amps[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, cutoff):
        amps[n] = amps[n - 1] * alpha / math.sqrt(float(n))
    total = float(np.linalg.norm(amps))
    if total > 0.0:
        amps /= total
    return amps


def squeezed_vacuum_amplitudes(r: float, theta: float, cutoff: int) -> np.ndarray:
    amps = np.zeros(cutoff, dtype=np.complex128)
    tanh_r = math.tanh(r)
    cosh_r = math.cosh(r)
    for n in range(0, cutoff, 2):
        k = n // 2
        coeff = math.sqrt(math.factorial(2 * k)) / ((2.0**k) * math.factorial(k) * math.sqrt(cosh_r))
        amps[n] = coeff * ((-np.exp(1j * theta) * tanh_r) ** k)
    total = float(np.linalg.norm(amps))
    if total > 0.0:
        amps /= total
    return amps


def gaussian_component_projected_fock_amplitudes(
    mean: np.ndarray, cov: np.ndarray, cutoff: int, branch_phase: float
) -> np.ndarray:
    channels = gaussian_projection_channels(mean, cov, cutoff, branch_phase)
    return np.asarray(channels["projected"], dtype=np.complex128)


def gaussian_projection_channels(
    mean: np.ndarray, cov: np.ndarray, cutoff: int, branch_phase: float
) -> dict[str, object]:
    alpha = complex(float(mean[0]), float(mean[1])) / math.sqrt(2.0)
    coherent = coherent_fock_amplitudes(alpha, cutoff)

    eigvals, eigvecs = covariance_geometry(cov)
    ratio = math.sqrt(max(float(eigvals[0]), 1e-12) / max(float(eigvals[1]), 1e-12))
    r = max(0.0, 0.5 * math.log(max(ratio, 1.0)))
    theta = math.atan2(float(eigvecs[1, 0]), float(eigvecs[0, 0]))
    squeezed = squeezed_vacuum_amplitudes(r, theta, cutoff)

    displacement_strength = abs(alpha)
    blend = displacement_strength / (displacement_strength + r + 1e-9)
    coherent_term = blend * coherent
    squeezed_term = (1.0 - blend) * squeezed
    amps = coherent_term + squeezed_term
    amps *= np.exp(1j * branch_phase)
    total = float(np.linalg.norm(amps))
    if total > 0.0:
        amps /= total
        coherent_term /= total
        squeezed_term /= total
    return {
        "alpha": alpha,
        "r": r,
        "theta": theta,
        "blend": blend,
        "coherent_term": coherent_term,
        "squeezed_term": squeezed_term,
        "projected": amps,
    }


def density_maxima() -> tuple[float, float]:
    q = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    p = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    q_grid, p_grid = np.meshgrid(q, p)

    mean_in, cov_in, mean_out, cov_out = gaussian_gate_example_data()
    gaussian_vmax = float(
        max(
            np.max(gaussian_density_grid(mean_in, cov_in, q_grid, p_grid)),
            np.max(gaussian_density_grid(mean_out, cov_out, q_grid, p_grid)),
        )
    )

    base_components, child_components = nongaussian_mixture_example_data()
    input_density = np.zeros_like(q_grid)
    for component in base_components:
        input_density += float(component["weight"]) * gaussian_density_grid(
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            q_grid,
            p_grid,
        )

    output_density = np.zeros_like(q_grid)
    for component in child_components:
        output_density += float(component["weight"]) * gaussian_density_grid(
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            q_grid,
            p_grid,
        )

    mixture_vmax = float(max(np.max(input_density), np.max(output_density)))
    return gaussian_vmax, mixture_vmax


def plot_gaussian_gate_example(
    output_dir: Path, cmap: LinearSegmentedColormap, shared_vmax: float
) -> Path:
    mean_in, cov_in, mean_out, cov_out = gaussian_gate_example_data()

    q = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    p = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    q_grid, p_grid = np.meshgrid(q, p)
    density_in = gaussian_density_grid(mean_in, cov_in, q_grid, p_grid)
    density_out = gaussian_density_grid(mean_out, cov_out, q_grid, p_grid)

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=2,
        nrows=1,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].imshow(
        density_in,
        extent=[q.min(), q.max(), p.min(), p.max()],
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=shared_vmax,
        interpolation="bilinear",
    )
    setup_phase_axis(axes[0], "Input Gaussian state")
    draw_covariance_ellipse(axes[0], mean_in, cov_in, edgecolor=PALETTE_DARK)

    im = axes[1].imshow(
        density_out,
        extent=[q.min(), q.max(), p.min(), p.max()],
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=shared_vmax,
        interpolation="bilinear",
    )
    setup_phase_axis(axes[1], "Output Gaussian state")
    draw_covariance_ellipse(
        axes[1],
        mean_in,
        cov_in,
        edgecolor=PALETTE_MID,
        linestyle="--",
        alpha=0.55,
    )
    draw_covariance_ellipse(axes[1], mean_out, cov_out, edgecolor=PALETTE_DARK)

    fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02, label="Density")

    output_path = output_dir / "gaussian_gate_on_gaussian_state.svg"
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_nongaussian_mixture_example(
    output_dir: Path, cmap: LinearSegmentedColormap, shared_vmax: float
) -> Path:
    base_components, child_components = nongaussian_mixture_example_data()

    q = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    p = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    q_grid, p_grid = np.meshgrid(q, p)

    input_density = np.zeros_like(q_grid)
    for component in base_components:
        input_density += float(component["weight"]) * gaussian_density_grid(
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            q_grid,
            p_grid,
        )

    output_density = np.zeros_like(q_grid)
    for component in child_components:
        output_density += float(component["weight"]) * gaussian_density_grid(
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            q_grid,
            p_grid,
        )

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=2,
        nrows=1,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].imshow(
        input_density,
        extent=[q.min(), q.max(), p.min(), p.max()],
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=shared_vmax,
        interpolation="bilinear",
    )
    setup_phase_axis(axes[0], "Input Gaussian mixture state")
    for component in base_components:
        draw_covariance_ellipse(
            axes[0],
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            edgecolor=str(component["color"]),
            linestyle=str(component["linestyle"]),
        )

    im = axes[1].imshow(
        output_density,
        extent=[q.min(), q.max(), p.min(), p.max()],
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=shared_vmax,
        interpolation="bilinear",
    )
    setup_phase_axis(axes[1], "Updated Gaussian mixture state")
    for component in base_components:
        draw_covariance_ellipse(
            axes[1],
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            edgecolor=str(component["color"]),
            linestyle="--",
            alpha=0.40,
            linewidth=0.75,
        )
    for component in child_components:
        draw_covariance_ellipse(
            axes[1],
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            edgecolor=str(component["color"]),
            linestyle=str(component["linestyle"]),
            alpha=0.40 + 0.70 * float(component["weight"]),
            linewidth=0.8,
        )

    fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02, label="Density")

    output_path = output_dir / "nongaussian_gate_on_gaussian_mixture.svg"
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_gaussian_mixture_projected_to_fock(
    output_dir: Path, cmap: LinearSegmentedColormap
) -> Path:
    components = gaussian_mixture_projection_example_data()
    cutoff = 12

    q = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    p = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    q_grid, p_grid = np.meshgrid(q, p)

    density = np.zeros_like(q_grid)
    weighted_amps: list[np.ndarray] = []
    for component in components:
        weight = float(component["weight"])
        density += weight * gaussian_density_grid(
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            q_grid,
            p_grid,
        )
        amps = gaussian_component_projected_fock_amplitudes(
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            cutoff,
            float(component["weight_phase"]),
        )
        weighted_amps.append(weight * amps)

    weighted_amp_stack = np.stack(weighted_amps, axis=0)
    total_amps = np.sum(weighted_amp_stack, axis=0)
    amp_norm = float(np.linalg.norm(total_amps))
    if amp_norm > 0.0:
        total_amps /= amp_norm
        weighted_amp_stack /= amp_norm

    amplitude = np.abs(total_amps)
    phase = np.angle(total_amps) / math.pi
    max_amp = float(np.max(amplitude))
    phase_mask = amplitude < max(0.035, 0.08 * max_amp)
    phase_display = phase.copy()
    phase_display[phase_mask] = 0.0
    unit_phase = np.ones_like(total_amps)
    nonzero_mask = amplitude > 1e-12
    unit_phase[nonzero_mask] = total_amps[nonzero_mask] / amplitude[nonzero_mask]
    component_contrib = np.real(weighted_amp_stack * np.conj(unit_phase))

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=3,
        nrows=1,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    im = axes[0].imshow(
        density,
        extent=[q.min(), q.max(), p.min(), p.max()],
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.max(density)),
        interpolation="bilinear",
    )
    setup_phase_axis(axes[0], "Gaussian mixture in phase space")
    for component in components:
        eigvals, eigvecs = covariance_geometry(np.asarray(component["cov"]))
        angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2.0 * math.sqrt(max(float(eigvals[0]), 1e-12))
        height = 2.0 * math.sqrt(max(float(eigvals[1]), 1e-12))
        axes[0].add_patch(
            Ellipse(
                xy=np.asarray(component["mean"]),
                width=width,
                height=height,
                angle=angle,
                facecolor=str(component["light_color"]),
                edgecolor="none",
                alpha=0.34,
                zorder=3,
            )
        )
        draw_covariance_ellipse(
            axes[0],
            np.asarray(component["mean"]),
            np.asarray(component["cov"]),
            edgecolor=str(component["color"]),
            linestyle=str(component["linestyle"]),
        )

    weight_ax = axes[0].inset_axes([0.55, 0.84, 0.34, 0.10])
    left = 0.0
    for component in components:
        weight = float(component["weight"])
        weight_ax.barh(
            [0.0],
            [weight],
            left=left,
            height=0.68,
            color=str(component["color"]),
            edgecolor="#FFF8EE",
            linewidth=0.55,
        )
        left += weight
    weight_ax.set_xlim(0.0, 1.0)
    weight_ax.set_ylim(-0.6, 0.6)
    weight_ax.set_xticks([0.0, 0.5, 1.0])
    weight_ax.set_yticks([])
    weight_ax.set_title("weights", fontsize=6.5, pad=1.0)
    weight_ax.tick_params(axis="x", labelsize=5.8, pad=1.0, length=2.0)
    for spine in weight_ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_edgecolor(GRID_COLOR)

    x = np.arange(cutoff)
    axes[1].axhline(0.0, color=GRID_COLOR, linewidth=0.65, alpha=0.50)
    for n in range(cutoff):
        negative_terms = []
        positive_terms = []
        for component_index, component in enumerate(components):
            value = float(component_contrib[component_index, n])
            if value < 0.0:
                negative_terms.append((component_index, component, value))
            else:
                positive_terms.append((component_index, component, value))

        running = 0.0
        for _, component, value in negative_terms:
            axes[1].bar(
                n,
                value,
                bottom=running,
                width=0.78,
                color=str(component["light_color"]),
                edgecolor="#FFF8EE",
                linewidth=0.45,
                alpha=0.92,
            )
            running += value
        for _, component, value in positive_terms:
            axes[1].bar(
                n,
                value,
                bottom=running,
                width=0.78,
                color=str(component["color"]),
                edgecolor="#FFF8EE",
                linewidth=0.45,
                alpha=0.88,
            )
            running += value
    axes[1].plot(x, amplitude, color=PROJECTION_DARK_COLORS[0], linewidth=0.95)
    axes[1].scatter(x, amplitude, color=PROJECTION_DARK_COLORS[0], s=12, zorder=3)
    axes[1].set_title("Projected Fock amplitude")
    axes[1].set_xlabel("Fock index $n$")
    axes[1].set_ylabel(r"Contribution to $|c_n|$")
    axes[1].set_xlim(-0.5, cutoff - 0.5)
    axes[1].set_ylim(
        min(-0.04 * max_amp, 1.20 * float(np.min(component_contrib))),
        1.08 * max_amp,
    )
    axes[1].set_xticks(range(cutoff))
    axes[1].grid(True, axis="y", linewidth=0.35, alpha=0.28, color=GRID_COLOR, linestyle="--")

    axes[2].axhline(0.0, color=GRID_COLOR, linewidth=0.65, alpha=0.50)
    axes[2].plot(x, phase_display, color=PROJECTION_DARK_COLORS[0], linewidth=1.0)
    axes[2].scatter(x[phase_mask], phase_display[phase_mask], color=PROJECTION_LIGHT_COLORS[1], s=12, zorder=3)
    axes[2].scatter(x[~phase_mask], phase_display[~phase_mask], color=PROJECTION_DARK_COLORS[1], s=16, zorder=4)
    axes[2].set_title("Projected Fock phase")
    axes[2].set_xlabel("Fock index $n$")
    axes[2].set_ylabel(r"$\arg(c_n)/\pi$")
    axes[2].set_xlim(-0.5, cutoff - 0.5)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_xticks(range(cutoff))
    axes[2].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axes[2].grid(True, axis="y", linewidth=0.35, alpha=0.28, color=GRID_COLOR, linestyle="--")

    fig.colorbar(im, ax=axes[0], fraction=0.050, pad=0.03, label="Density")

    output_path = output_dir / "gaussian_mixture_projected_to_fock.svg"
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_single_gaussian_projected_to_fock(
    output_dir: Path, cmap: LinearSegmentedColormap
) -> Path:
    mean = np.array([-0.72, 0.88], dtype=np.float64)
    cov = np.array([[0.18, 0.06], [0.06, 0.34]], dtype=np.float64)
    branch_phase = 0.46
    cutoff = 12

    q = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    p = np.linspace(AXIS_MIN, AXIS_MAX, 240)
    q_grid, p_grid = np.meshgrid(q, p)
    density = gaussian_density_grid(mean, cov, q_grid, p_grid)

    channels = gaussian_projection_channels(mean, cov, cutoff, branch_phase)
    coherent_term = np.asarray(channels["coherent_term"], dtype=np.complex128)
    squeezed_term = np.asarray(channels["squeezed_term"], dtype=np.complex128)
    projected = np.asarray(channels["projected"], dtype=np.complex128)
    alpha = complex(channels["alpha"])
    blend = float(channels["blend"])
    r = float(channels["r"])
    theta = float(channels["theta"])

    amplitude = np.abs(projected)
    phase = np.angle(projected) / math.pi
    max_amp = float(np.max(amplitude))
    phase_mask = amplitude < max(0.035, 0.08 * max_amp)
    phase_display = phase.copy()
    phase_display[phase_mask] = 0.0

    coherent_mag = np.abs(coherent_term)
    squeezed_mag = np.abs(squeezed_term)
    eigvals, eigvecs = covariance_geometry(cov)
    major = math.sqrt(max(float(eigvals[0]), 1e-12)) * eigvecs[:, 0]
    minor = math.sqrt(max(float(eigvals[1]), 1e-12)) * eigvecs[:, 1]

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=3,
        nrows=2,
        panel_aspect=0.88,
    )
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.16, 0.98, 0.98], height_ratios=[1.0, 1.0])
    ax_state = fig.add_subplot(gs[:, 0])
    ax_disp = fig.add_subplot(gs[0, 1])
    ax_sq = fig.add_subplot(gs[1, 1])
    ax_amp = fig.add_subplot(gs[0, 2])
    ax_phase = fig.add_subplot(gs[1, 2])

    im = ax_state.imshow(
        density,
        extent=[q.min(), q.max(), p.min(), p.max()],
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.max(density)),
        interpolation="bilinear",
    )
    setup_phase_axis(ax_state, "1. Gaussian state in phase space")
    ax_state.add_patch(
        Ellipse(
            xy=mean,
            width=2.0 * math.sqrt(max(float(eigvals[0]), 1e-12)),
            height=2.0 * math.sqrt(max(float(eigvals[1]), 1e-12)),
            angle=math.degrees(math.atan2(float(eigvecs[1, 0]), float(eigvecs[0, 0]))),
            facecolor=PROJECTION_LIGHT_COLORS[0],
            edgecolor="none",
            alpha=0.34,
            zorder=3,
        )
    )
    draw_covariance_ellipse(ax_state, mean, cov, edgecolor=PROJECTION_DARK_COLORS[0])
    ax_state.annotate(
        "",
        xy=(float(mean[0]), float(mean[1])),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=PROJECTION_DARK_COLORS[1]),
    )
    ax_state.text(
        0.16,
        0.60,
        r"$d$",
        transform=ax_state.transAxes,
        color=PROJECTION_DARK_COLORS[1],
        fontsize=8.0,
    )
    ax_state.plot(
        [mean[0] - major[0], mean[0] + major[0]],
        [mean[1] - major[1], mean[1] + major[1]],
        color=PROJECTION_DARK_COLORS[2],
        linewidth=1.0,
        alpha=0.9,
    )
    ax_state.plot(
        [mean[0] - minor[0], mean[0] + minor[0]],
        [mean[1] - minor[1], mean[1] + minor[1]],
        color=PROJECTION_DARK_COLORS[2],
        linewidth=0.85,
        alpha=0.7,
    )
    ax_state.text(
        0.04,
        0.96,
        r"$d \rightarrow \alpha = (q+ip)/\sqrt{2}$" "\n" r"$\Sigma \rightarrow (r,\theta)$ from eig$(\Sigma)$",
        transform=ax_state.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=GRID_COLOR,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFF8EE", edgecolor="none", alpha=0.85),
    )

    x = np.arange(cutoff)
    ax_disp.bar(
        x,
        coherent_mag,
        width=0.72,
        color=PROJECTION_DARK_COLORS[0],
        edgecolor="#FFF8EE",
        linewidth=0.45,
        alpha=0.90,
    )
    ax_disp.set_title("2a. Template from displacement")
    ax_disp.set_xlabel("Fock index $n$")
    ax_disp.set_ylabel(r"$| \langle n|\alpha \rangle |$")
    ax_disp.set_xlim(-0.5, cutoff - 0.5)
    ax_disp.set_xticks(range(cutoff))
    ax_disp.grid(True, axis="y", linewidth=0.35, alpha=0.28, color=GRID_COLOR, linestyle="--")
    ax_disp.text(
        0.04,
        0.95,
        rf"$|\alpha|={abs(alpha):.2f}$" "\n" rf"$\lambda={blend:.2f}$",
        transform=ax_disp.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=GRID_COLOR,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFF8EE", edgecolor="none", alpha=0.82),
    )

    ax_sq.bar(
        x,
        squeezed_mag,
        width=0.72,
        color=PROJECTION_DARK_COLORS[2],
        edgecolor="#FFF8EE",
        linewidth=0.45,
        alpha=0.90,
    )
    ax_sq.set_title("2b. Template from covariance")
    ax_sq.set_xlabel("Fock index $n$")
    ax_sq.set_ylabel(r"$| \langle n|S(r,\theta)|0\rangle |$")
    ax_sq.set_xlim(-0.5, cutoff - 0.5)
    ax_sq.set_xticks(range(cutoff))
    ax_sq.grid(True, axis="y", linewidth=0.35, alpha=0.28, color=GRID_COLOR, linestyle="--")
    ax_sq.text(
        0.04,
        0.95,
        (
            rf"$r={r:.2f}$" "\n"
            rf"$\theta/\pi={theta / math.pi:.2f}$" "\n"
            r"odd $n$ stay near $0$"
        ),
        transform=ax_sq.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=GRID_COLOR,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFF8EE", edgecolor="none", alpha=0.84),
    )

    ax_amp.bar(
        x,
        amplitude,
        width=0.72,
        color=PROJECTION_DARK_COLORS[1],
        edgecolor="#FFF8EE",
        linewidth=0.45,
        alpha=0.92,
    )
    ax_amp.plot(x, amplitude, color=PROJECTION_DARK_COLORS[0], linewidth=0.95)
    ax_amp.scatter(x, amplitude, color=PROJECTION_DARK_COLORS[0], s=12, zorder=3)
    ax_amp.set_title("3. Projected Fock amplitude")
    ax_amp.set_xlabel("Fock index $n$")
    ax_amp.set_ylabel(r"$|c_n|$")
    ax_amp.set_xlim(-0.5, cutoff - 0.5)
    ax_amp.set_ylim(0.0, 1.08 * max_amp)
    ax_amp.set_xticks(range(cutoff))
    ax_amp.grid(True, axis="y", linewidth=0.35, alpha=0.28, color=GRID_COLOR, linestyle="--")
    ax_amp.text(
        0.04,
        0.95,
        r"$c_n \approx e^{i\phi}\!\left[\lambda \langle n|\alpha\rangle + (1-\lambda)\langle n|S(r,\theta)|0\rangle\right]$",
        transform=ax_amp.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=GRID_COLOR,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFF8EE", edgecolor="none", alpha=0.82),
    )

    ax_phase.axhline(0.0, color=GRID_COLOR, linewidth=0.65, alpha=0.50)
    ax_phase.plot(x, phase_display, color=PROJECTION_DARK_COLORS[0], linewidth=1.0)
    ax_phase.scatter(x[phase_mask], phase_display[phase_mask], color=PROJECTION_LIGHT_COLORS[1], s=12, zorder=3)
    ax_phase.scatter(x[~phase_mask], phase_display[~phase_mask], color=PROJECTION_DARK_COLORS[1], s=16, zorder=4)
    ax_phase.set_title("4. Projected Fock phase")
    ax_phase.set_xlabel("Fock index $n$")
    ax_phase.set_ylabel(r"$\arg(c_n)/\pi$")
    ax_phase.set_xlim(-0.5, cutoff - 0.5)
    ax_phase.set_ylim(-1.05, 1.05)
    ax_phase.set_xticks(range(cutoff))
    ax_phase.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_phase.grid(True, axis="y", linewidth=0.35, alpha=0.28, color=GRID_COLOR, linestyle="--")

    for artist in [
        ConnectionPatch(
            xyA=(0.98, 0.70),
            coordsA=ax_state.transAxes,
            xyB=(0.02, 0.80),
            coordsB=ax_disp.transAxes,
            arrowstyle="->",
            lw=1.0,
            color=GRID_COLOR,
            alpha=0.75,
            mutation_scale=11,
        ),
        ConnectionPatch(
            xyA=(0.98, 0.32),
            coordsA=ax_state.transAxes,
            xyB=(0.02, 0.74),
            coordsB=ax_sq.transAxes,
            arrowstyle="->",
            lw=1.0,
            color=GRID_COLOR,
            alpha=0.75,
            mutation_scale=11,
        ),
        ConnectionPatch(
            xyA=(0.98, 0.58),
            coordsA=ax_disp.transAxes,
            xyB=(0.02, 0.82),
            coordsB=ax_amp.transAxes,
            arrowstyle="->",
            lw=1.0,
            color=GRID_COLOR,
            alpha=0.75,
            mutation_scale=11,
        ),
        ConnectionPatch(
            xyA=(0.98, 0.54),
            coordsA=ax_sq.transAxes,
            xyB=(0.02, 0.70),
            coordsB=ax_amp.transAxes,
            arrowstyle="->",
            lw=1.0,
            color=GRID_COLOR,
            alpha=0.75,
            mutation_scale=11,
        ),
    ]:
        fig.add_artist(artist)

    fig.colorbar(im, ax=ax_state, fraction=0.050, pad=0.03, label="Density")

    output_path = output_dir / "gaussian_state_projected_to_fock_process.svg"
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    cmap = build_colormap()
    gaussian_vmax, mixture_vmax = density_maxima()
    shared_vmax = max(gaussian_vmax, mixture_vmax)
    saved = [
        plot_gaussian_gate_example(output_dir, cmap, shared_vmax),
        plot_nongaussian_mixture_example(output_dir, cmap, shared_vmax),
        plot_gaussian_mixture_projected_to_fock(output_dir, cmap),
        plot_single_gaussian_projected_to_fock(output_dir, cmap),
    ]
    for path in saved:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
