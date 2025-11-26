from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr


def plot_mean_ndvi(
    baseline: xr.DataArray,
    target: xr.DataArray,
    anomaly: xr.DataArray,
    output_path: str | None = None,
) -> None:
    """
    Plot baseline, target and anomaly NDVI maps side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    baseline.plot(ax=axes[0], cmap="YlGn", vmin=0, vmax=1)
    axes[0].set_title("Baseline mean NDVI")
    axes[0].set_axis_off()

    target.plot(ax=axes[1], cmap="YlGn", vmin=0, vmax=1)
    axes[1].set_title("Target mean NDVI")
    axes[1].set_axis_off()

    anomaly.plot(ax=axes[2], cmap="RdBu", vmin=-0.3, vmax=0.3)
    axes[2].set_title("NDVI anomaly (target - baseline)")
    axes[2].set_axis_off()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_forest_anomaly(
    anomaly_forest: xr.DataArray,
    output_path: str | None = None,
) -> None:
    """
    Plot NDVI anomaly restricted to forest pixels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
    anomaly_forest.plot(ax=ax, cmap="RdBu", vmin=-0.3, vmax=0.3)
    ax.set_title("Forest-only NDVI anomaly")
    ax.set_axis_off()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
