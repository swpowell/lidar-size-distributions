# SGP Workflow

This directory now contains the canonical SGP eddy-size workflow:

- `generate_event_tables.py`: build regular and resampled per-day event tables
- `build_curated_dataset.py`: combine event tables into one analysis-ready dataset
- `run_h1_analysis.py`: upper-tail eddy size versus PBL depth
- `run_h2_analysis.py`: upper-tail eddy size versus height inside the PBL
- `run_h3_analysis.py`: cloudy versus clear secondary analysis

Core package modules:

- `config.py`: centralized path and daylight defaults
- `detection.py`: lidar processing, QC, and eddy detection
- `generation.py`: canonical event-table generation
- `curation.py`: combined dataset construction
- `labels.py`: daylight and cloud labels
- `clouds.py`: METAR/cloud-condition parsing helpers
- `hypotheses.py`: H1/H2/H3 summaries and plots

Typical flow:

1. Run `python SGP/generate_event_tables.py ...`
2. Run `python SGP/build_curated_dataset.py ...`
3. Run `python SGP/run_h1_analysis.py ...`
4. Run `python SGP/run_h2_analysis.py ...`
5. Optionally run `python SGP/run_h3_analysis.py ...`
