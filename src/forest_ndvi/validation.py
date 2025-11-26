from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def _flatten_valid(
    arr: xr.DataArray,
    mask: Optional[xr.DataArray] = None,
) -> np.ndarray:
    """
    Flatten a DataArray to a 1D numpy array, optionally applying a boolean mask
    and removing NaNs.
    """
    data = arr
    if mask is not None:
        data = data.where(mask)

    vals = data.values.astype("float32").ravel()
    vals = vals[~np.isnan(vals)]
    return vals


def plot_ndvi_histograms(
    baseline_mean: xr.DataArray,
    target_mean: xr.DataArray,
    forest_mask: xr.DataArray,
    output_path: Path,
) -> None:
    """
    Plot histograms of baseline and target mean NDVI over forest pixels.
    """
    baseline_vals = _flatten_valid(baseline_mean, forest_mask)
    target_vals = _flatten_valid(target_mean, forest_mask)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)

    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(
        baseline_vals,
        bins=bins,
        alpha=0.6,
        label="Baseline mean NDVI",
        density=True,
    )
    ax.hist(
        target_vals,
        bins=bins,
        alpha=0.6,
        label="Target mean NDVI",
        density=True,
    )

    ax.set_xlabel("NDVI")
    ax.set_ylabel("Density")
    ax.set_title("Forest mean NDVI distributions\n(baseline vs target)")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_histogram(
    anomaly: xr.DataArray,
    forest_mask: xr.DataArray,
    output_path: Path,
) -> None:
    """
    Plot a histogram of NDVI anomalies over forest pixels.
    """
    anomaly_vals = _flatten_valid(anomaly, forest_mask)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)

    bins = np.linspace(-0.4, 0.4, 41)
    ax.hist(
        anomaly_vals,
        bins=bins,
        alpha=0.8,
        density=True,
    )

    ax.set_xlabel("NDVI anomaly (target - baseline)")
    ax.set_ylabel("Density")
    ax.set_title("Forest NDVI anomaly distribution")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_forest_mask_overlay(
    baseline_mean: xr.DataArray,
    forest_mask: xr.DataArray,
    output_path: Path,
) -> None:
    """
    Overlay the forest mask contours on top of the baseline mean NDVI map.

    Useful to visually check that the WorldCover forest class aligns with
    high-NDVI areas and that the reprojection is sensible.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=150)

    im = baseline_mean.plot(
        ax=ax,
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
        add_colorbar=True,
    )
    ax.set_title("Baseline mean NDVI with forest mask overlay")
    ax.set_axis_off()

    forest_bool = forest_mask.astype(int)
    forest_bool.plot.contour(
        ax=ax,
        levels=[0.5],
        colors="red",
        linewidths=0.6,
        add_colorbar=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ndvi_scatter(
    baseline_mean: xr.DataArray,
    target_mean: xr.DataArray,
    forest_mask: xr.DataArray,
    output_path: Path,
) -> None:
    """
    Scatter/hexbin plot of baseline vs target NDVI over forest pixels.

    Uses a joint mask so that baseline and target vectors have the same length.
    """
    # Joint validity mask: forest pixel + both NDVI fields finite
    joint_mask = (
        forest_mask
        & baseline_mean.notnull()
        & target_mean.notnull()
    )

    b = baseline_mean.where(joint_mask).values.astype("float32").ravel()
    t = target_mean.where(joint_mask).values.astype("float32").ravel()

    valid = np.isfinite(b) & np.isfinite(t)
    b = b[valid]
    t = t[valid]

    if b.size == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)

    hb = ax.hexbin(
        b,
        t,
        gridsize=80,
        cmap="viridis",
        mincnt=1,
    )
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.0, label="1:1 line")

    ax.set_xlabel("Baseline mean NDVI")
    ax.set_ylabel("Target mean NDVI")
    ax.set_title("Forest NDVI: baseline vs target")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")

    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Pixel count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_true_color_quicklook(
    ds: xr.Dataset,
    output_path: Path,
    title: str = "Sentinel-2 true colour quicklook",
) -> None:
    """
    Create a simple true-colour RGB quicklook from a Sentinel-2 Dataset.

    Uses bands B04 (R), B03 (G), B02 (B), averaged over time, with a
    simple percentile-based contrast stretch.
    """
    # Mean over time for each band
    r = ds["B04"].mean(dim="time").values.astype("float32")
    g = ds["B03"].mean(dim="time").values.astype("float32")
    b = ds["B02"].mean(dim="time").values.astype("float32")

    rgb = np.stack([r, g, b], axis=-1)

    def stretch(channel: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(channel[np.isfinite(channel)], [2, 98])
        stretched = (channel - lo) / (hi - hi + 1e-6)  # bug: hi-hi; fix below?
        return np.clip(stretched, 0.0, 1.0)

    # Fix the stretch bug correctly:
    def stretch(channel: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(channel[np.isfinite(channel)], [2, 98])
        stretched = (channel - lo) / (hi - lo + 1e-6)
        return np.clip(stretched, 0.0, 1.0)

    rgb_stretched = np.zeros_like(rgb, dtype="float32")
    for i in range(3):
        rgb_stretched[..., i] = stretch(rgb[..., i])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.imshow(rgb_stretched)
    ax.set_title(title)
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
