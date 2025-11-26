from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class NDVIConfig:
    """
    Configuration for the NDVI anomaly pipeline.
    """

    # Path to GeoJSON file defining the Area Of Interest
    aoi_path: Path

    # Baseline period (start, end) for NDVI climatology
    baseline_dates: Tuple[str, str] = ("2024-06-01", "2024-06-30")

    # Target period (start, end) to compare against the baseline
    target_dates: Tuple[str, str] = ("2025-06-01", "2025-06-30")

    # Sentinel-2 collection name on Planetary Computer
    collection: str = "sentinel-2-l2a"

    # Maximum allowed cloud cover (%) when selecting Sentinel-2 scenes
    cloud_cover_max: float = 40.0

    # UTM EPSG code used for reprojection and gridding
    utm_epsg: int = 32631

    # Spatial resolution in metres (Sentinel-2 native is 10 m)
    resolution: int = 20

    # Small epsilon to avoid division by zero in NDVI computation
    ndvi_eps: float = 1e-6

    # ESA WorldCover collection name on Planetary Computer
    worldcover_collection: str = "esa-worldcover"

    # WorldCover product year to use
    worldcover_year: int = 2021

    # WorldCover class code corresponding to forest / tree cover
    forest_class: int = 10

    # Spatial chunk size (pixels) for x/y when loading with odc-stac
    chunks_xy: int = 512

    # Temporal chunk size (number of time steps per chunk)
    chunks_time: int = 2

    @property
    def crs(self) -> str:
        """Return the CRS string derived from the UTM EPSG code."""
        return f"EPSG:{self.utm_epsg}"
