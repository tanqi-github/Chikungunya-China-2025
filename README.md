# Chikungunya in China, 2025: public figure code

This repository contains the public plotting code accompanying the city-week
dataset describing reported chikungunya cases in China in 2025.

The script reconstructs the five descriptive figures using only the released
aggregate files. It does not read case-level records or the internal data
construction, classification, review, or validation files.

## Public inputs

Place the following released files in one local directory:

- `chikv_china2025_city_week.csv`
- `city_lookup.csv`
- `release_summary.json`

The city-week table contains the reporting-city code, ISO week, reported case
count, and inferred imported case count. The script calculates inferred local
cases as:

```text
reported_cases - inferred_imported_cases
```

It then reconstructs the city milestones and weekly spatial concentration
measures used in the figures.

## Basemap

The maps use the province- and prefecture-level administrative boundary files
published by Tianditu (National Platform for Common Geospatial Information
Services), map approval number **GS(2024)0650**.

Download the administrative division GeoJSON files from:

<https://cloudcenter.tianditu.gov.cn/administrativeDivision>

Keep these two original filenames in one local directory:

- `中国_省.geojson`
- `中国_市.geojson`

The basemap files are not redistributed in this repository.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the virtual environment using the command appropriate for your
operating system.

## Usage

```bash
python scripts/plot_scidata_public_release.py \
  --release-dir /path/to/public_release \
  --basemap-dir /path/to/tianditu_geojson \
  --output-dir /path/to/public_figures
```

On Windows PowerShell:

```powershell
python scripts/plot_scidata_public_release.py `
  --release-dir "C:\path\to\public_release" `
  --basemap-dir "C:\path\to\tianditu_geojson" `
  --output-dir "C:\path\to\public_figures"
```

Alternatively, set `SCIDATA_TIANDITU_DIR` instead of passing
`--basemap-dir`. The script writes SVG and PDF files with editable text and
600 dpi PNG files.

## Reconstructed outputs

- `fig1_weekly_curve`: weekly epidemic curve, ten leading reporting cities,
  and inferred local cases in Guangdong versus other provinces
- `fig2_first_timing`: cumulative reported cases and city-level local-case
  milestones
- `fig3_expansion`: accumulation of reporting cities and conditional milestone
  intervals
- `fig4_concentration`: weekly spatial concentration of reported cases
- `fig5_source_composition`: inferred local and imported case composition by
  reporting province and week

The figures describe reported surveillance patterns. Reporting city is not
necessarily the place of infection or residence, and the milestone summaries
should not be interpreted as confirmed transmission routes or biological
transmission speed.

## Reproducibility checks

Before plotting, the script checks the release summary against the aggregate
tables, validates non-negative counts and a complete city-week grid, confirms
the city lookup, and verifies that all reporting-city codes match the supplied
basemap. Random point jitter in the milestone interval figure uses a fixed
seed.

## Data availability

The released aggregate dataset and its persistent repository identifier will
be linked here when the data deposit is public.

