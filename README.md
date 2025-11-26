# Forest NDVI Anomaly – Sentinel-2 & ESA WorldCover

This repository contains a small but complete Python library, `forest_ndvi`, for
computing NDVI and NDVI anomalies over forested areas using Sentinel-2 L2A and
ESA WorldCover data via Microsoft Planetary Computer.

The project demonstrates:

- Design of a **reproducible ETL pipeline** for multi-temporal EO data.
- Use of **STAC** (via `pystac-client` and `odc-stac`) to discover and load Sentinel-2 scenes.
- Application of **cloud / quality masking** using the Sentinel-2 Scene Classification Layer (SCL).
- Computation of NDVI time series and **baseline vs target period anomalies**.
- Restriction of the analysis to **forest pixels** using ESA WorldCover tree-cover classes.
- Generation of **summary statistics** over forest NDVI anomalies.
- Basic **QA / validation**:
  - NDVI and anomaly histograms over forest pixels.
  - Baseline vs target NDVI scatter/hexbin plot.
  - Overlay of the WorldCover forest mask on mean NDVI.
  - True-colour Sentinel-2 quicklooks for baseline and target periods.
- A clean `src/` package layout with a high-level `run_ndvi_anomaly_pipeline` API.

  <img width="2211" height="712" alt="ndvi_mean_and_anomaly" src="https://github.com/user-attachments/assets/dd1b7896-0b7b-44dd-8c29-91a00863e509" />

---

## Usage

Create and activate a virtual environment, install dependencies, and run the pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py
```

---

## Acknowledgement

If you found these demonstrations useful, please consider contacting the author at eleni.tsaprazi@gmail.com (Eleni Tsaprazi) to discuss a way to acknowledge the contribution. Feedback is welcome!
