"""
forest_ndvi
===========

Small library for computing NDVI and NDVI anomalies over forested areas
using Sentinel-2 and ESA WorldCover data via Microsoft Planetary Computer.
"""

from .config import NDVIConfig
from .pipeline import run_ndvi_anomaly_pipeline

__all__ = ["NDVIConfig", "run_ndvi_anomaly_pipeline"]
