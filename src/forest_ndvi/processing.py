from __future__ import annotations

from typing import List

import numpy as np
import xarray as xr
import odc.stac as odc

from .config import NDVIConfig


# Sentinel-2 Scene Classification codes to treat as valid land pixels
VALID_SCL = [4, 5, 6, 7, 11]  # vegetation + bare soil, tweak if needed


def load_s2_stack(items: List, cfg: NDVIConfig) -> xr.Dataset:
    """
    Load Sentinel-2 items into an xarray.Dataset using odc-stac.

    Includes RGB (B02, B03, B04), NIR (B08) and SCL for masking.
    """
    bands = ["B02", "B03", "B04", "B08", "SCL"]

    ds = odc.load(
        items,
        bands=bands,
        chunks={"x": cfg.chunks_xy, "y": cfg.chunks_xy, "time": cfg.chunks_time},
        crs=cfg.crs,
        resolution=cfg.resolution,
    )
    return ds


def mask_valid_s2(ds: xr.Dataset) -> xr.Dataset:
    """
    Mask out invalid pixels based on Sentinel-2 SCL band.

    Parameters
    ----------
    ds : xarray.Dataset
        Sentinel-2 dataset including SCL band.

    Returns
    -------
    ds_masked : xarray.Dataset
        Dataset with invalid pixels set to NaN.
    """
    scl = ds["SCL"]
    valid = xr.zeros_like(scl, dtype=bool)
    for v in VALID_SCL:
        valid = valid | (scl == v)
    return ds.where(valid)


def add_ndvi(ds: xr.Dataset, eps: float) -> xr.Dataset:
    """
    Compute NDVI from Sentinel-2 red (B04) and NIR (B08) bands.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with B04 and B08 bands.
    eps : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    ds_ndvi : xarray.Dataset
        Dataset with an added 'ndvi' DataArray.
    """
    nir = ds["B08"].astype("float32")
    red = ds["B04"].astype("float32")
    ndvi = (nir - red) / (nir + red + eps)
    ds = ds.assign(ndvi=ndvi)
    return ds


def load_worldcover_forest_mask(
    items: List,
    cfg: NDVIConfig,
    template: xr.DataArray,
) -> xr.DataArray:
    """
    Load ESA WorldCover and build a boolean forest mask aligned to a template grid.

    We load WorldCover in the same CRS/resolution as Sentinel-2 and then
    align it to the template's x/y coordinates via nearest-neighbour interpolation.
    """
    wc = odc.load(
        items,
        bands=["map"],
        chunks={"x": cfg.chunks_xy * 2, "y": cfg.chunks_xy * 2},
        crs=cfg.crs,
        resolution=cfg.resolution,
    ).squeeze("time")

    wc_map = wc["map"]

    # Align to template grid via nearest-neighbour interpolation
    wc_aligned = wc_map.interp(
        x=template["x"],
        y=template["y"],
        method="nearest",
    )

    forest_mask = wc_aligned == cfg.forest_class
    return forest_mask


def compute_mean_ndvi(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute mean NDVI over time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with 'ndvi' DataArray.

    Returns
    -------
    mean_ndvi : xarray.DataArray
        Mean NDVI over time.
    """
    return ds["ndvi"].mean(dim="time")


def compute_anomaly(
    baseline_mean: xr.DataArray,
    target_mean: xr.DataArray,
) -> xr.DataArray:
    """
    Compute NDVI anomaly (target - baseline).

    Parameters
    ----------
    baseline_mean : xarray.DataArray
        Mean NDVI over baseline period.
    target_mean : xarray.DataArray
        Mean NDVI over target period.

    Returns
    -------
    anomaly : xarray.DataArray
        NDVI anomaly.
    """
    return target_mean - baseline_mean


def summarise_forest_anomaly(
    anomaly: xr.DataArray,
    forest_mask: xr.DataArray,
) -> dict:
    """
    Compute basic statistics of NDVI anomaly over forest pixels.

    Parameters
    ----------
    anomaly : xarray.DataArray
        NDVI anomaly map.
    forest_mask : xarray.DataArray
        Boolean mask selecting forest pixels.

    Returns
    -------
    stats : dict
        Dictionary with mean, std and number of pixels.
    """
    masked = anomaly.where(forest_mask)
    vals = masked.values.astype("float32").ravel()
    vals = vals[~np.isnan(vals)]

    if vals.size == 0:
        return {"mean": np.nan, "std": np.nan, "n_pixels": 0}

    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "n_pixels": int(vals.size),
    }
