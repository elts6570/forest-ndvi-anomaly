from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.geometry.base import BaseGeometry


def load_aoi(aoi_path: Path) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float]]:
    """
    Load AOI geometry and bounding box from a GeoJSON file.

    Parameters
    ----------
    aoi_path : Path
        Path to a GeoJSON file containing a polygon AOI.

    Returns
    -------
    geom : shapely.geometry.base.BaseGeometry
        AOI geometry in EPSG:4326.
    bbox : tuple
        Bounding box (minx, miny, maxx, maxy).
    """
    gdf = gpd.read_file(aoi_path)

    # Ensure we have a CRS; default to WGS84 if missing
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    bbox = tuple(gdf.total_bounds)  # (minx, miny, maxx, maxy)
    return gdf, bbox

