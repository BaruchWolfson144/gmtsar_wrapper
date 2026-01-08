# GMTSAR InSAR Processing Pipeline - Usage Guide

## Overview

This Python wrapper provides a modular interface for running the complete GMTSAR SBAS (Small Baseline Subset) InSAR time series processing workflow. The pipeline consists of 8 main processing stages, from initial setup to phase unwrapping. see the .pdf file for original manual.

## Quick Start

### Option 1: Using a Configuration File (Recommended)

1. Copy and customize the example configuration:
```bash
cp config_example.yaml my_project_config.yaml
# Edit my_project_config.yaml with your parameters
```

2. Run the complete pipeline:
```bash
python main.py my_project_config.yaml --sequential
```

### Option 2: Command-Line Arguments

Run individual stages with command-line arguments:
```bash
# Stage 1: Create directory structure
python main.py /path/to/project --stage 1 --orbit asc

# Stage 2: Create DEM
python main.py /path/to/project --stage 2 \
  --minlon -157 --maxlon -154.2 --minlat 18 --maxlat 20.4 --dem-mode 1

# Stage 4: Download orbits
python main.py /path/to/project --stage 4 --orbit-mode 1

# And so on...
```

## Processing Stages

### Stage 1: Directory Structure Creation
Creates the standard GMTSAR directory tree with subdirectories for data, orbits, SLC images, interferograms, etc.

**Required Parameters:**
- `project_root`: Path to project directory
- `orbit`: Orbit type ("asc" or "des")

**Example:**
```bash
python main.py /data/hawaii_insar --stage 1 --orbit asc
```

### Stage 2: DEM Creation
Downloads and prepares Digital Elevation Model for the area of interest.

**Required Parameters:**
- `minlon`, `maxlon`, `minlat`, `maxlat`: Bounding box coordinates
- `dem-mode`: 1 for SRTM1 (30m), 2 for SRTM3 (90m)

**Example:**
```bash
python main.py /data/hawaii_insar --stage 2 \
  --minlon -157 --maxlon -154.2 --minlat 18 --maxlat 20.4 --dem-mode 1
```

### Stage 4: Orbit Download (and Optional Reframing)
Downloads precise or restituted orbit files. Optionally performs frame stitching/cropping.

**Required Parameters:**
- `orbit-mode`: 1 for precise, 2 for restituted orbits

**Optional Reframing Parameters:**
- `--reframe`: Enable reframing
- `pin1-lon`, `pin1-lat`: First pin coordinates
- `pin2-lon`, `pin2-lat`: Second pin coordinates

**Example (simple orbit download):**
```bash
python main.py /data/hawaii_insar --stage 4 --orbit-mode 1
```

**Example (with reframing):**
```bash
python main.py /data/hawaii_insar --stage 4 --orbit-mode 1 --reframe \
  --pin1-lon -157 --pin1-lat 18 --pin2-lon -154.2 --pin2-lat 20.4
```

### Stage 5: Master Selection and Alignment
Preprocesses subswaths and automatically selects optimal master image based on baseline analysis.

**Parameters:**
- `subswath-list`: List of subswaths to process (default: F1, F2, F3)

**Example:**
```bash
python main.py /data/hawaii_insar --stage 5 --subswath-list F1 F2 F3
```

**Note:** The master image is automatically selected and stored for use in subsequent stages.

### Stage 6: Interferogram Generation
Creates interferograms for all valid image pairs within baseline thresholds.

**Parameters:**
- `threshold-time`: Temporal baseline threshold in days (default: 100)
- `threshold-baseline`: Perpendicular baseline threshold in meters (default: 150)
- `intf-config`: Optional path to intf.in configuration file

**Example:**
```bash
python main.py /data/hawaii_insar --stage 6 \
  --threshold-time 100 --threshold-baseline 150
```

**Note:** This stage uses the master image automatically selected in Stage 5.

### Stage 7: Interferogram Merging
Merges interferograms across multiple subswaths.

**Parameters:**
- `merge-mode`: 1 for merge all, 2 for merge specified subswaths (default: 1)

**Example:**
```bash
python main.py /data/hawaii_insar --stage 7 --merge-mode 1
```

### Stage 8: Phase Unwrapping
Unwraps interferometric phase with quality control masks.

**Parameters:**
- `coherence-threshold`: Coherence threshold (default: 0.075)
- `corr-threshold`: Correlation threshold (default: 0.01)
- `max-dis-threshold`: Maximum discontinuity threshold (default: 40)
- `use-landmask`: Enable landmask (requires landmask.grd file)
- `use-mask-def`: Use default masking (default: true)

**Example:**
```bash
python main.py /data/hawaii_insar --stage 8 \
  --coherence-threshold 0.075 --corr-threshold 0.01 --max-dis-threshold 40
```

## Execution Modes

### Sequential Execution
Run all stages in order, automatically passing parameters between stages:

```bash
python main.py config.yaml --sequential
```

**Features:**
- Automatically skips already-completed stages
- Passes discovered parameters (like master image) between stages
- Saves state after each stage for resume capability

### Individual Stage Execution
Run a specific stage only:

```bash
python main.py /path/to/project --stage 5
```

**When to use:**
- Re-running a single stage with different parameters
- Debugging a specific stage
- Continuing after manual intervention

### Resume from Last Stage
Continue processing from the last completed stage:

```bash
python main.py /path/to/project --resume
```

**Use cases:**
- Processing was interrupted
- Previous run failed at a specific stage
- Want to continue with remaining stages

### Check Processing Status
View current processing state and completed stages:

```bash
python main.py /path/to/project --status
```

**Output includes:**
- List of completed stages
- Stored parameters and outputs
- Last update timestamp

## State Management

The pipeline maintains processing state in:
```
<project_root>/wrapper_meta/state/project_state.json
```

This file stores:
- Completed stages
- Parameters used in each stage
- Outputs from each stage (e.g., master image selection)
- Timestamps

**Benefits:**
- Resume capability after interruption
- Parameters discovered in early stages are available to later stages
- Full processing history and traceability

## Configuration Files

Configuration files can be in YAML or JSON format. Use them to:
- Define all parameters in one place
- Share processing configurations
- Version control your processing workflows
- Ensure reproducibility

**Example YAML structure:**
```yaml
project_root: "/path/to/project"
orbit: "asc"

dem:
  minlon: -157.0
  maxlon: -154.2
  minlat: 18.0
  maxlat: 20.4
  mode: 1

interferograms:
  threshold_time: 100
  threshold_baseline: 150

# ... more parameters
```

See [config_example.yaml](config_example.yaml) and [config_example.json](config_example.json) for complete examples.

## Common Workflows

### First-Time Complete Processing
```bash
# 1. Prepare configuration
cp config_example.yaml hawaii_2018.yaml
# Edit hawaii_2018.yaml with your parameters

# 2. Run complete pipeline
python main.py hawaii_2018.yaml --sequential

# 3. Check status
python main.py /path/to/project --status
```

### Re-processing From Specific Stage
```bash
# Re-run interferogram generation with different thresholds
python main.py /path/to/project --stage 6 \
  --threshold-time 120 --threshold-baseline 200
```

### Processing With Manual Intervention
```bash
# Run stages 1-5
python main.py config.yaml --sequential --stop-at 5

# Manually inspect master selection results
# Review: <project_root>/wrapper_meta/logs/preproc_*.json

# Continue with remaining stages
python main.py /path/to/project --resume
```

## Troubleshooting

### Check Logs
Processing logs are stored in:
```
<project_root>/wrapper_meta/logs/
```

Each stage creates its own log file with:
- Commands executed
- Standard output and error
- Timestamps
- Success/failure status

### Common Issues

**Issue:** "Master image not found in state"
- **Cause:** Stage 6+ requires master from Stage 5
- **Solution:** Run Stage 5 first, or check state file for master parameter

**Issue:** "SAFE_filelist not found"
- **Cause:** No .SAFE data files in data directory
- **Solution:** Ensure Sentinel-1 data is downloaded to `<project_root>/<orbit>/data/`

**Issue:** "DEM file not found"
- **Cause:** Stage 2 not completed or failed
- **Solution:** Check Stage 2 logs and re-run if needed

### Reset Processing
To start fresh and clear all state:
```bash
rm <project_root>/wrapper_meta/state/project_state.json
```

**Warning:** This will lose all processing history and discovered parameters.

## Advanced Usage

### Custom GMTSAR Parameters
Each stage script (e.g., `run_interferograms_06.py`) can be run independently with its own parameters. See individual script documentation for advanced options.

### Parallel Processing
Currently, stages run sequentially. Future versions may support parallel processing of multiple interferogram pairs.

### Integration with Jupyter Notebooks
The pipeline classes can be imported and used in notebooks:

```python
from pathlib import Path
from main import InSARPipeline

# Create pipeline
pipeline = InSARPipeline("/path/to/project", "config.yaml")

# Run specific stage
outputs = pipeline.stage_05_preprocess_subswaths()

# Access state
master = pipeline.state.get_parameter("master")
print(f"Selected master: {master}")
```

## Reference Documentation

- **GMTSAR Tutorial:** `sentinel_time_series.pdf`
- **GMTSAR Website:** https://topex.ucsd.edu/gmtsar/
- **Individual Stage Scripts:** `make_dir_tree_01.py`, `stage_02_download_data.py`, `make_dem_03.py`, etc.

## Support

For issues or questions:
1. Check processing logs in `wrapper_meta/logs/`
2. Review state file: `wrapper_meta/state/project_state.json`
3. Consult GMTSAR documentation
4. Refer to `sentinel_time_series.pdf` tutorial

## Version Information

- Wrapper Version: 0.1
- Supported GMTSAR Version: 6.0+
- Python Version: 3.8+
- Tested Platforms: Linux (Ubuntu 20.04+, CentOS 7+)
