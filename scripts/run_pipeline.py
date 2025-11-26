from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forest_ndvi import NDVIConfig, run_ndvi_anomaly_pipeline


def main() -> None:
    aoi_path = REPO_ROOT / "data" / "aoi.geojson"
    output_dir = REPO_ROOT / "outputs"

    cfg = NDVIConfig(
        aoi_path=aoi_path,
        # tweak UTM zone etc. here if needed
    )

    stats, _ = run_ndvi_anomaly_pipeline(cfg, output_dir=output_dir)
    print("Forest NDVI anomaly stats:", stats)


if __name__ == "__main__":
    main()
