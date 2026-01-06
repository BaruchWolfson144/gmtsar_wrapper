# GMTSAR Wrapper - Workflow Changes Summary

## Overview
This document describes the refactoring performed to eliminate code duplication between `make_dem_02.py` and `make_orbits_04.py`, ensuring each script has a single, well-defined responsibility aligned with the GMTSAR processing workflow.

## Problem Identified
Both scripts contained duplicate reframing functions (`create_pins_file`, `reframe_data`, `run_reframe`), creating confusion about which script to use and violating the single responsibility principle.

## Solution Based on GMTSAR Manual

According to `sentinel_time_series.pdf`, the correct processing order is:

1. **Step 2** (Page 3): DEM Creation → `make_dem_02.py`
2. **Step 3a-3b** (Pages 4-8): Data Download (handled separately)
3. **Step 3c** (Page 9): Create pins.ll for reframing
4. **Step 4** (Pages 11-12): Download orbits ± reframing
   - **4a**: Simple orbit download only
   - **4b**: Combined orbit download + reframing (RECOMMENDED)

### Key Insight from Manual
The `organize_files_tops.csh` script (used in Step 4b) performs **BOTH** orbit downloading **AND** reframing when provided with both `SAFE_filelist` and `pins.ll`. This is the recommended approach.

## Refactoring Changes

### make_dem_02.py - NOW HANDLES
✅ **DEM Creation Only (Step 2)**
- Creates dem.grd using `make_dem.csh`
- Links DEM to project directories (F1/F2/F3/topo, merge, des)
- Logs DEM creation metadata

**Removed:**
- ❌ All reframing functions (`create_pins_file`, `reframe_data`, `run_reframe`)
- ❌ Reframe command-line arguments
- ❌ Reframe workflow logic

**Usage:**
```bash
python make_dem_02.py /path/to/project \
    --minlon -157 --maxlon -154.2 \
    --minlat 18 --maxlat 20.4 \
    --mode 1
```

### make_orbits_04.py - NOW HANDLES
✅ **Orbit Download + Optional Reframing (Steps 3c & 4)**

**Functions:**
1. `create_orbit_list()` - Creates SAFE_filelist
2. `create_pins_file()` - Creates pins.ll for reframing (Step 3c)
3. `download_orbits_simple()` - Simple orbit download (Section 4a)
4. `download_orbits_with_reframe()` - Combined workflow (Section 4b, RECOMMENDED)
5. `run_download_orbits()` - Main orchestration function

**Two Workflow Options:**

**Option 1: Simple orbit download (no reframing)**
```bash
python make_orbits_04.py /path/to/project --mode 1
```

**Option 2: Combined orbit download + reframing (RECOMMENDED)**
```bash
python make_orbits_04.py /path/to/project --mode 1 \
    --reframe \
    --pin1_lon -157 --pin1_lat 18 \
    --pin2_lon -154.2 --pin2_lat 20.4
```

## Complete Processing Sequence

```bash
# Step 1: Create project structure (separate script)
python make_dirs_01.py /path/to/project

# Step 2: Create DEM (MUST RUN FIRST)
python make_dem_02.py /path/to/project \
    --minlon -157 --maxlon -154.2 \
    --minlat 18 --maxlat 20.4 \
    --mode 1

# Step 3a-3b: Download Sentinel-1 data (separate script/manual)

# Step 3c & 4: Download orbits with reframing
python make_orbits_04.py /path/to/project --mode 1 \
    --reframe \
    --pin1_lon -157 --pin1_lat 18.5 \
    --pin2_lon -154.2 --pin2_lat 19.8

# Step 5+: Continue with alignment, interferogram generation, etc.
```

## Benefits of This Refactoring

1. **Clear Separation of Concerns**
   - `make_dem_02.py`: DEM creation only
   - `make_orbits_04.py`: Orbits + reframing

2. **Follows GMTSAR Manual Structure**
   - Matches the order described in `sentinel_time_series.pdf`
   - Implements recommended workflow (Section 4b)

3. **No Code Duplication**
   - Reframing functions exist in only one place
   - Easier to maintain and debug

4. **User-Friendly**
   - Clear command-line arguments
   - Descriptive help messages
   - Better error messages

5. **Flexible**
   - Can skip reframing if not needed
   - Supports both ascending and descending orbits
   - Comprehensive logging

## Technical Details

### Reframing Process (Section 4b)
When `--reframe` is specified, the script:

1. Creates `SAFE_filelist` from .SAFE files in data directory
2. Creates `pins.ll` file in `{orbit}/reframed/` directory
3. Runs `organize_files_tops_linux.csh` mode 1 (~5 min)
   - Downloads precise/restituted orbits
   - Prepares files
4. Runs `organize_files_tops_linux.csh` mode 2 (~35 min)
   - Creates stitched/cropped SAFE directories (F*_F* pattern)
5. Logs all commands and results

### Pin Coordinates
- **Ascending orbits**: Pin 1 (south) → Pin 2 (north)
- **Descending orbits**: Pin 1 (north) → Pin 2 (south)
- Use Google Earth with .kml files to determine pin locations

## Migration Guide

If you have existing scripts calling the old versions:

**Old approach (DON'T USE):**
```bash
python make_dem_02.py /path/to/project \
    --minlon X --maxlon Y --minlat A --maxlat B \
    --reframe --pin1_lon P1 --pin1_lat Q1 --pin2_lon P2 --pin2_lat Q2
```

**New approach (USE THIS):**
```bash
# First: DEM only
python make_dem_02.py /path/to/project \
    --minlon X --maxlon Y --minlat A --maxlat B

# Then: Orbits + reframing
python make_orbits_04.py /path/to/project \
    --reframe --pin1_lon P1 --pin1_lat Q1 --pin2_lon P2 --pin2_lat Q2
```

## References
- `sentinel_time_series.pdf` - Complete GMTSAR SBAS manual
  - Section 2 (Page 3): DEM preparation
  - Section 3c (Page 9): Reframing setup
  - Section 4a (Page 11): Simple orbit download
  - Section 4b (Pages 11-12): Combined orbit + reframing workflow
