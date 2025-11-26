from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import xarray as xr

from .config import NDVIConfig
from .aoi import load_aoi
from .stac_io import search_sentinel2_items, search_worldcover_items
from .processing import (
    load_s2_stack,
    mask_valid_s2,
    add_ndvi,
    load_worldcover_forest_mask,
    compute_mean_ndvi,
    compute_anomaly,
    summarise_forest_anomaly,
)
from .plotting import plot_mean_ndvi, plot_forest_anomaly
from .validation import (
    plot_ndvi_histograms,
    plot_anomaly_histogram,
    plot_forest_mask_overlay,
    plot_ndvi_scatter,
    plot_true_color_quicklook,
)


def run_ndvi_anomaly_pipeline(
    cfg: NDVIConfig,
    output_dir: Path,
) -> Tuple[Dict[str, float], xr.DataArray]:
    """
    Run the full NDVI anomaly pipeline and save plots and validation outputs
    to output_dir.

    Returns
    -------
    stats : dict
        Summary statistics over forest NDVI anomaly.
    anomaly_forest : xarray.DataArray
        NDVI anomaly restricted to forest pixels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load AOI geometry and bounding box
    aoi_gdf, bbox = load_aoi(cfg.aoi_path)

    # Search Sentinel-2 items for baseline and target periods
    baseline_items = search_sentinel2_items(cfg, bbox, cfg.baseline_dates)
    target_items = search_sentinel2_items(cfg, bbox, cfg.target_dates)

    # Load Sentinel-2 stacks (includes RGB + NIR + SCL)
    baseline_ds = load_s2_stack(baseline_items, cfg)
    target_ds = load_s2_stack(target_items, cfg)

    # Apply SCL mask and compute NDVI
    baseline_clean = add_ndvi(mask_valid_s2(baseline_ds), cfg.ndvi_eps)
    target_clean = add_ndvi(mask_valid_s2(target_ds), cfg.ndvi_eps)

    # Compute mean NDVI (this triggers computation)
    baseline_mean = compute_mean_ndvi(baseline_clean).compute()
    target_mean = compute_mean_ndvi(target_clean).compute()

    # NDVI anomaly
    anomaly = compute_anomaly(baseline_mean, target_mean).compute()

    # WorldCover forest mask on the same grid
    wc_items = search_worldcover_items(cfg, bbox)
    forest_mask = load_worldcover_forest_mask(
        wc_items,
        cfg,
        template=baseline_mean,
    ).compute()

    anomaly_forest = anomaly.where(forest_mask)

    # Summary statistics over forest pixels
    stats = summarise_forest_anomaly(anomaly, forest_mask)

    # Main maps
    plot_mean_ndvi(
        baseline_mean,
        target_mean,
        anomaly,
        output_path=output_dir / "ndvi_mean_and_anomaly.png",
    )
    plot_forest_anomaly(
        anomaly_forest,
        output_path=output_dir / "ndvi_forest_anomaly.png",
    )

    # Validation plots: distributions
    plot_ndvi_histograms(
        baseline_mean,
        target_mean,
        forest_mask,
        output_path=output_dir / "ndvi_histograms_forest.png",
    )
    plot_anomaly_histogram(
        anomaly,
        forest_mask,
        output_path=output_dir / "ndvi_anomaly_histogram_forest.png",
    )

    # Validation plots: spatial overlay and scatter
    plot_forest_mask_overlay(
        baseline_mean,
        forest_mask,
        output_path=output_dir / "ndvi_baseline_forest_overlay.png",
    )
    plot_ndvi_scatter(
        baseline_mean,
        target_mean,
        forest_mask,
        output_path=output_dir / "ndvi_baseline_vs_target_scatter.png",
    )

    # True-colour quicklooks for baseline and target periods
    plot_true_color_quicklook(
        baseline_ds,
        output_path=output_dir / "s2_true_color_baseline.png",
        title="Sentinel-2 true colour (baseline period)",
    )
    plot_true_color_quicklook(
        target_ds,
        output_path=output_dir / "s2_true_color_target.png",
        title="Sentinel-2 true colour (target period)",
    )

    # Save anomaly field to NetCDF
    anomaly_forest.name = "ndvi_anomaly_forest"
    anomaly_forest.to_netcdf(output_dir / "ndvi_anomaly_forest.nc")

    return stats, anomaly_forest
