#!/usr/bin/env python3
"""
================================================================================
GMTSAR Stage 09: SBAS (Small BAseline Subset) Analysis================================================================================

This module implements the SBAS time series analysis stage following the
GMTSAR Sentinel-1 tutorial (Section 10).

SBAS performs a least-squares inversion of unwrapped interferograms to produce:
- Velocity map (vel.grd) - mm/yr LOS velocity
- Displacement time series (disp_*.grd) - mm displacement at each epoch
- RMS residuals (rms.grd) - quality metric
- DEM error estimate (dem_err.grd) - optional

WORKFLOW (from PDF manual Section 10):
1. Prepare input files:
   - scene.tab: scene_id and days since reference
   - intf.tab: paths to unwrap.grd, corr.grd, scene IDs, baselines
2. Run SBAS inversion
3. Generate visualizations

STATUS: Production - integrated into main.py
Author: Claude Code Assistant
Date: 2026-02-11
================================================================================
"""

import argparse
import subprocess
import json
import re
import os
import logging
from pathlib import Path
import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving to files
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_cmd(cmd: str, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """
    Execute a shell command and return (returncode, stdout, stderr).

    Preserves the full environment including PATH for GMTSAR tools.
    """
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_baseline_table(baseline_file: Path) -> Dict[str, Dict]:
    """
    Parse baseline_table.dat to extract scene information.

    Format: scene_stem decimal_year day_of_year parallel_baseline perpendicular_baseline
    Example: S1_20180201_ALL_F1 2018031.1874165505 1491 19.088688436880 56.434825714407

    Returns:
        Dict mapping scene_stem to {decimal_year, day, parallel_bl, perpendicular_bl}
    """
    scenes = {}
    with open(baseline_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                stem = parts[0]
                decimal_year = float(parts[1])
                day = int(parts[2])
                parallel_bl = float(parts[3])
                perpendicular_bl = float(parts[4])

                # Extract YYYYDOY from decimal_year (first 7 chars)
                scene_id = str(int(decimal_year))[:7]

                scenes[stem] = {
                    "scene_id": scene_id,
                    "decimal_year": decimal_year,
                    "day": day,
                    "parallel_baseline": parallel_bl,
                    "perpendicular_baseline": perpendicular_bl,
                }
    return scenes


def parse_intf_in(intf_in_file: Path) -> List[Tuple[str, str]]:
    """
    Parse intf.in file to get interferogram pairs.

    Format: scene1:scene2
    Example: S1_20180201_ALL_F1:S1_20180213_ALL_F1

    Returns:
        List of (ref_scene, sec_scene) tuples
    """
    pairs = []
    with open(intf_in_file) as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
    return pairs


def get_grid_dimensions(grd_file: Path) -> Tuple[int, int]:
    """
    Get x and y dimensions of a GMT grid file using gmt grdinfo.

    Returns:
        (xdim, ydim) tuple
    """
    cmd = f"gmt grdinfo -C {grd_file}"
    rc, out, err = run_cmd(cmd)

    if rc != 0:
        raise RuntimeError(f"gmt grdinfo failed: {err}")

    # Output format: file xmin xmax ymin ymax zmin zmax dx dy nx ny ...
    parts = out.strip().split()
    if len(parts) >= 11:
        nx = int(parts[9])  # n_columns
        ny = int(parts[10])  # n_rows
        return nx, ny

    raise RuntimeError(f"Could not parse grdinfo output: {out}")


def get_radar_parameters(prm_file: Path) -> Dict[str, float]:
    """
    Extract radar parameters from a PRM file.

    Parameters extracted:
    - radar_wavelength (m)
    - rng_samp_rate (Hz)
    - near_range (m)

    Returns:
        Dict with radar parameters
    """
    params = {}

    with open(prm_file) as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key == "radar_wavelength":
                    params["wavelength"] = float(value)
                elif key == "rng_samp_rate":
                    params["rng_samp_rate"] = float(value)
                elif key == "near_range":
                    params["near_range"] = float(value)

    return params


def calculate_range_distance(prm_file: Path, x_min: float, x_max: float) -> float:
    """
    Calculate range distance to center of interferogram.

    Formula from PDF manual:
    Range = ({[(speed_of_light) / (rng_samp_rate) / 2] * ((x_min+x_max)/2)} / 2) + near_range

    Args:
        prm_file: Path to master PRM file
        x_min: Minimum x coordinate from grid
        x_max: Maximum x coordinate from grid

    Returns:
        Range distance in meters
    """
    params = get_radar_parameters(prm_file)

    speed_of_light = 299792458.0  # m/s
    rng_samp_rate = params.get("rng_samp_rate", 64345238.125714)
    near_range = params.get("near_range", 800000.0)

    x_center = (x_min + x_max) / 2
    range_dist = (((speed_of_light / rng_samp_rate) / 2) * x_center / 2) + near_range

    return range_dist


# =============================================================================
# SBAS INPUT PREPARATION
# =============================================================================

def prepare_scene_tab(
    baseline_table: Path,
    output_dir: Path
) -> Tuple[Path, int]:
    """
    Prepare scene.tab file for SBAS.

    Format: <scene_id> <days>
    Example:
        2018031 1491
        2018037 1497

    Scene IDs must be in chronological order!

    Args:
        baseline_table: Path to baseline_table.dat
        output_dir: Directory to write scene.tab

    Returns:
        (path_to_scene_tab, num_scenes)
    """
    scenes = parse_baseline_table(baseline_table)

    # Sort by day number (chronological order)
    sorted_scenes = sorted(scenes.values(), key=lambda x: x["day"])

    scene_tab = output_dir / "scene.tab"
    with open(scene_tab, "w") as f:
        for scene in sorted_scenes:
            f.write(f"{scene['scene_id']} {scene['day']}\n")

    return scene_tab, len(sorted_scenes)


def prepare_intf_tab(
    intf_in: Path,
    baseline_table: Path,
    merge_dir: Path,
    output_dir: Path,
    unwrap_file: str = "unwrap.grd",
    corr_file: str = "corr.grd"
) -> Tuple[Path, int]:
    """
    Prepare intf.tab file for SBAS.

    Format: <path_to_unwrap.grd> <path_to_corr.grd> <ref_scene_id> <sec_scene_id> <b_perp_diff>
    Example:
        ../merge/2018031_2018037/unwrap.grd ../merge/2018031_2018037/corr.grd 2018031 2018037 -97

    Args:
        intf_in: Path to intf.in file with interferogram pairs
        baseline_table: Path to baseline_table.dat
        merge_dir: Path to merge directory containing interferograms
        output_dir: Directory to write intf.tab
        unwrap_file: Name of unwrapped phase file (default: unwrap.grd)
        corr_file: Name of correlation file (default: corr.grd)

    Returns:
        (path_to_intf_tab, num_interferograms)
    """
    pairs = parse_intf_in(intf_in)
    scenes = parse_baseline_table(baseline_table)

    intf_tab = output_dir / "intf.tab"
    valid_count = 0

    with open(intf_tab, "w") as f:
        for ref_stem, sec_stem in pairs:
            if ref_stem not in scenes or sec_stem not in scenes:
                logger.warning(f"Skipping pair {ref_stem}:{sec_stem} - not in baseline_table")
                continue

            ref_info = scenes[ref_stem]
            sec_info = scenes[sec_stem]

            # Calculate perpendicular baseline difference
            b_perp_diff = sec_info["perpendicular_baseline"] - ref_info["perpendicular_baseline"]

            # Build interferogram directory name (YYYYDOY_YYYYDOY format)
            ref_id = ref_info["scene_id"]
            sec_id = sec_info["scene_id"]
            intf_dir = f"{ref_id}_{sec_id}"

            # Build paths relative to SBAS directory
            unwrap_path = f"../merge/{intf_dir}/{unwrap_file}"
            corr_path = f"../merge/{intf_dir}/{corr_file}"

            # Check if files exist
            full_unwrap = merge_dir / intf_dir / unwrap_file
            full_corr = merge_dir / intf_dir / corr_file

            if not full_unwrap.exists():
                logger.warning(f"{full_unwrap} not found, skipping")
                continue
            if not full_corr.exists():
                logger.warning(f"{full_corr} not found, skipping")
                continue

            # Write line (baseline as integer per SBAS requirement)
            f.write(f"{unwrap_path} {corr_path} {ref_id} {sec_id} {int(b_perp_diff)}\n")
            valid_count += 1

    return intf_tab, valid_count


def run_prep_sbas(
    project_root: Path,
    orbit: str,
    subswath: str = "F1",
    unwrap_file: str = "unwrap.grd",
    corr_file: str = "corr.grd"
) -> Dict[str, Any]:
    """
    Prepare all input files for SBAS using the GMTSAR prep_sbas.csh script.

    This is an alternative to manual preparation that uses the built-in script.

    Args:
        project_root: Project root directory
        orbit: Orbit directory (asc/des)
        subswath: Subswath to use for baseline_table.dat (default: F1)
        unwrap_file: Name of unwrapped phase file
        corr_file: Name of correlation file

    Returns:
        Dict with preparation results
    """
    sbas_dir = project_root / orbit / "SBAS"
    sbas_dir.mkdir(parents=True, exist_ok=True)

    merge_dir = project_root / orbit / "merge"
    intf_in = project_root / orbit / subswath / "intf.in"
    baseline_table = project_root / orbit / subswath / "baseline_table.dat"

    # Copy intf.in and baseline_table.dat to SBAS directory
    import shutil
    shutil.copy(intf_in, sbas_dir / "intf.in")
    shutil.copy(baseline_table, sbas_dir / "baseline_table.dat")

    # Run prep_sbas.csh
    cmd = f"prep_sbas.csh intf.in baseline_table.dat ../merge {unwrap_file} {corr_file}"
    rc, out, err = run_cmd(cmd, cwd=sbas_dir)

    result = {
        "command": cmd,
        "return_code": rc,
        "stdout": out,
        "stderr": err,
        "sbas_dir": str(sbas_dir),
    }

    # Check for output files
    scene_tab = sbas_dir / "scene.tab"
    intf_tab = sbas_dir / "intf.tab"

    if scene_tab.exists():
        result["scene_tab"] = str(scene_tab)
        with open(scene_tab) as f:
            result["num_scenes"] = sum(1 for _ in f)

    if intf_tab.exists():
        result["intf_tab"] = str(intf_tab)
        with open(intf_tab) as f:
            result["num_interferograms"] = sum(1 for _ in f)

    return result


# =============================================================================
# SBAS EXECUTION
# =============================================================================

def run_sbas_inversion(
    sbas_dir: Path,
    intf_tab: Path,
    scene_tab: Path,
    num_intfs: int,
    num_scenes: int,
    xdim: int,
    ydim: int,
    wavelength: float = 0.0554658,
    incidence: float = 37.0,
    range_dist: float = 900000.0,
    smooth: float = 5.0,
    compute_rms: bool = True,
    compute_dem_err: bool = True,
    atm_iterations: int = 0,
    use_mmap: bool = True
) -> Dict[str, Any]:
    """
    Run SBAS inversion.

    Command format from PDF:
    sbas intf.tab scene.tab N S xdim ydim [-atm ni] [-smooth sf]
         [-wavelength wl] [-incidence inc] [-range rng] [-rms] [-dem] [-mmap]

    Args:
        sbas_dir: SBAS working directory
        intf_tab: Path to intf.tab
        scene_tab: Path to scene.tab
        num_intfs: Number of interferograms (N)
        num_scenes: Number of scenes (S)
        xdim: X dimension of interferograms
        ydim: Y dimension of interferograms
        wavelength: Radar wavelength in meters (default: Sentinel-1 C-band)
        incidence: Incidence angle in degrees (default: 37)
        range_dist: Range distance in meters
        smooth: Smoothing factor (default: 5.0)
        compute_rms: Whether to compute RMS grid (default: True)
        compute_dem_err: Whether to compute DEM error grid (default: True)
        atm_iterations: Number of atmospheric iterations (0 = none)
        use_mmap: Whether to use memory-mapped I/O to reduce RAM usage (default: True)

    Returns:
        Dict with SBAS results
    """
    # Build SBAS command
    cmd_parts = [
        "sbas",
        str(intf_tab.name),
        str(scene_tab.name),
        str(num_intfs),
        str(num_scenes),
        str(xdim),
        str(ydim),
    ]

    # Add optional parameters
    if atm_iterations > 0:
        cmd_parts.extend(["-atm", str(atm_iterations)])

    cmd_parts.extend(["-smooth", str(smooth)])
    cmd_parts.extend(["-wavelength", str(wavelength)])
    cmd_parts.extend(["-incidence", str(incidence)])
    cmd_parts.extend(["-range", str(int(range_dist))])

    if compute_rms:
        cmd_parts.append("-rms")

    if compute_dem_err:
        cmd_parts.append("-dem")

    if use_mmap:
        cmd_parts.append("-mmap")

    cmd = " ".join(cmd_parts)

    logger.info(f"Running SBAS command:")
    logger.info(f"  {cmd}")
    logger.info(f"  Working directory: {sbas_dir}")

    # Run SBAS (can take a while)
    rc, out, err = run_cmd(cmd, cwd=sbas_dir)

    result = {
        "command": cmd,
        "return_code": rc,
        "stdout": out[-2000:] if len(out) > 2000 else out,  # Truncate long output
        "stderr": err[-2000:] if len(err) > 2000 else err,
        "parameters": {
            "num_interferograms": num_intfs,
            "num_scenes": num_scenes,
            "xdim": xdim,
            "ydim": ydim,
            "wavelength": wavelength,
            "incidence": incidence,
            "range": range_dist,
            "smooth": smooth,
        },
    }

    # Check for output files
    output_files = {
        "vel.grd": "LOS velocity (mm/yr)",
        "rms.grd": "RMS residuals (mm)",
        "dem_err.grd": "DEM error (m)",
    }

    result["outputs"] = {}
    for fname, description in output_files.items():
        fpath = sbas_dir / fname
        if fpath.exists():
            result["outputs"][fname] = {
                "path": str(fpath),
                "description": description,
                "exists": True,
            }
        else:
            result["outputs"][fname] = {"exists": False}

    # Check for displacement grids
    disp_grids = list(sbas_dir.glob("disp_*.grd"))
    result["displacement_grids"] = {
        "count": len(disp_grids),
        "files": [str(f.name) for f in sorted(disp_grids)[:10]],  # First 10
    }

    return result


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_baseline_network_plot(
    intf_tab: Path,
    scene_tab: Path,
    output_path: Path
) -> bool:
    """
    Create a baseline network plot showing temporal vs perpendicular baselines.

    This visualization shows:
    - X-axis: Time (date)
    - Y-axis: Perpendicular baseline (m)
    - Lines connecting interferometric pairs
    - Dots for each SAR acquisition

    Args:
        intf_tab: Path to intf.tab
        scene_tab: Path to scene.tab
        output_path: Path to save the plot

    Returns:
        True if successful, False otherwise
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        logger.warning("matplotlib/numpy not available, skipping baseline network plot")
        return False

    try:
        # Parse scene.tab
        scenes = {}
        with open(scene_tab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    scene_id = parts[0]
                    day = int(parts[1])
                    scenes[scene_id] = day

        # Parse intf.tab for pairs and baselines
        pairs = []
        baselines = {}

        with open(intf_tab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    ref_id = parts[2]
                    sec_id = parts[3]
                    b_perp = float(parts[4])

                    pairs.append((ref_id, sec_id))

                    # Accumulate baselines (approximate)
                    if ref_id not in baselines:
                        baselines[ref_id] = 0.0
                    if sec_id not in baselines:
                        baselines[sec_id] = baselines[ref_id] + b_perp

        # Convert days to dates (assuming reference is 2014-01-01 for Sentinel-1)
        reference_date = datetime.date(2014, 1, 1)

        scene_dates = {}
        scene_baselines = {}
        for scene_id, day in scenes.items():
            scene_dates[scene_id] = reference_date + datetime.timedelta(days=day)
            scene_baselines[scene_id] = baselines.get(scene_id, 0.0)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot interferogram connections
        for ref_id, sec_id in pairs:
            if ref_id in scene_dates and sec_id in scene_dates:
                x = [scene_dates[ref_id], scene_dates[sec_id]]
                y = [scene_baselines[ref_id], scene_baselines[sec_id]]
                ax.plot(x, y, 'b-', alpha=0.3, linewidth=0.5)

        # Plot acquisition points
        dates = list(scene_dates.values())
        bls = [scene_baselines[sid] for sid in scene_dates.keys()]
        ax.scatter(dates, bls, c='red', s=30, zorder=5, label='SAR acquisitions')

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Perpendicular Baseline (m)', fontsize=12)
        ax.set_title(f'SBAS Baseline Network\n{len(scenes)} scenes, {len(pairs)} interferograms', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved baseline network plot to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating baseline network plot: {e}")
        return False


def create_connectivity_histogram(
    intf_tab: Path,
    scene_tab: Path,
    output_path: Path
) -> bool:
    """
    Create histogram showing number of connections per scene.

    Good SBAS networks should have similar connectivity across all scenes.

    Args:
        intf_tab: Path to intf.tab
        scene_tab: Path to scene.tab
        output_path: Path to save the plot

    Returns:
        True if successful, False otherwise
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        logger.warning("matplotlib/numpy not available, skipping connectivity histogram")
        return False

    try:
        # Count connections per scene
        connections = {}

        with open(intf_tab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    ref_id = parts[2]
                    sec_id = parts[3]

                    connections[ref_id] = connections.get(ref_id, 0) + 1
                    connections[sec_id] = connections.get(sec_id, 0) + 1

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of connections per scene
        scene_ids = sorted(connections.keys())
        conn_counts = [connections[sid] for sid in scene_ids]

        ax1.bar(range(len(scene_ids)), conn_counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Scene Index', fontsize=12)
        ax1.set_ylabel('Number of Connections', fontsize=12)
        ax1.set_title('Interferogram Connections per Scene', fontsize=14)
        ax1.axhline(y=np.mean(conn_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(conn_counts):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram of connection distribution
        ax2.hist(conn_counts, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Connections', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Scene Connectivity', fontsize=14)
        ax2.axvline(x=np.mean(conn_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(conn_counts):.1f}')
        ax2.axvline(x=np.min(conn_counts), color='orange', linestyle=':',
                    label=f'Min: {np.min(conn_counts)}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved connectivity histogram to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating connectivity histogram: {e}")
        return False


def create_temporal_baseline_histogram(
    intf_tab: Path,
    scene_tab: Path,
    output_path: Path
) -> bool:
    """
    Create histogram of temporal baselines in the SBAS network.

    Args:
        intf_tab: Path to intf.tab
        scene_tab: Path to scene.tab
        output_path: Path to save the plot

    Returns:
        True if successful, False otherwise
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        logger.warning("matplotlib/numpy not available, skipping temporal baseline histogram")
        return False

    try:
        # Parse scene.tab for days
        scenes = {}
        with open(scene_tab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    scenes[parts[0]] = int(parts[1])

        # Calculate temporal baselines
        temporal_baselines = []
        perp_baselines = []

        with open(intf_tab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    ref_id = parts[2]
                    sec_id = parts[3]
                    b_perp = abs(float(parts[4]))

                    if ref_id in scenes and sec_id in scenes:
                        dt = abs(scenes[sec_id] - scenes[ref_id])
                        temporal_baselines.append(dt)
                        perp_baselines.append(b_perp)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Temporal baseline histogram
        ax1.hist(temporal_baselines, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Temporal Baseline (days)', fontsize=12)
        ax1.set_ylabel('Number of Interferograms', fontsize=12)
        ax1.set_title(f'Temporal Baseline Distribution\nMean: {np.mean(temporal_baselines):.0f} days', fontsize=14)
        ax1.axvline(x=np.mean(temporal_baselines), color='red', linestyle='--')
        ax1.grid(True, alpha=0.3)

        # Perpendicular baseline histogram
        ax2.hist(perp_baselines, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Perpendicular Baseline (m)', fontsize=12)
        ax2.set_ylabel('Number of Interferograms', fontsize=12)
        ax2.set_title(f'Perpendicular Baseline Distribution\nMean: {np.mean(perp_baselines):.0f} m', fontsize=14)
        ax2.axvline(x=np.mean(perp_baselines), color='red', linestyle='--')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved baseline histograms to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating temporal baseline histogram: {e}")
        return False


# =============================================================================
# METADATA LOGGING
# =============================================================================

def write_meta_log(
    project_root: Path,
    orbit: str,
    prep_info: Dict,
    sbas_info: Dict,
    viz_info: Dict
) -> Path:
    """
    Write metadata log for Stage 09.

    Args:
        project_root: Project root directory
        orbit: Orbit directory name
        prep_info: Input preparation results
        sbas_info: SBAS execution results
        viz_info: Visualization results

    Returns:
        Path to log file
    """
    meta = {
        "step": 9,
        "stage_name": "SBAS Analysis",
        "orbit": orbit,
        "timestamp": datetime.datetime.now().isoformat(),
        "preparation": prep_info,
        "sbas_inversion": sbas_info,
        "visualizations": viz_info,
    }

    log_dir = project_root / "wrapper_meta" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"step9_{orbit}_sbas.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return log_path


# =============================================================================
# MAIN SBAS FUNCTION
# =============================================================================

def run_sbas(
    project_root: Path,
    orbit: str = "asc",
    subswath: str = "F1",
    unwrap_file: str = "unwrap.grd",
    corr_file: str = "corr.grd",
    smooth: float = 5.0,
    atm_iterations: int = 0,
    use_prep_script: bool = True,
    use_mmap: bool = True
) -> Tuple[Path, str]:
    """
    Run complete SBAS analysis pipeline.

    This function:
    1. Creates SBAS directory
    2. Prepares input files (scene.tab, intf.tab)
    3. Extracts radar parameters
    4. Runs SBAS inversion
    5. Generates visualizations
    6. Writes metadata log

    Args:
        project_root: Project root directory
        orbit: Orbit directory (asc/des)
        subswath: Subswath for baseline_table.dat (default: F1)
        unwrap_file: Name of unwrapped phase file
        corr_file: Name of correlation file
        smooth: Smoothing factor for SBAS
        atm_iterations: Number of atmospheric iterations
        use_prep_script: Whether to use prep_sbas.csh (True) or manual prep (False)
        use_mmap: Whether to use memory-mapped I/O to reduce RAM usage (default: True)

    Returns:
        (log_path, result_message)
    """
    result_msg = ""

    # Create SBAS directory
    sbas_dir = project_root / orbit / "SBAS"
    sbas_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"SBAS directory: {sbas_dir}")

    merge_dir = project_root / orbit / "merge"
    intf_in = project_root / orbit / subswath / "intf.in"
    baseline_table = project_root / orbit / subswath / "baseline_table.dat"

    # =================================================================
    # STEP 1: Prepare input files (prep_sbas.csh or manual)
    # =================================================================
    logger.info("-" * 50)
    if use_prep_script:
        logger.info("STEP 1: Preparing input files (prep_sbas.csh)")
    else:
        logger.info("STEP 1: Preparing input files (manual scene.tab + intf.tab)")
    logger.info("-" * 50)

    if use_prep_script:
        # Use GMTSAR's prep_sbas.csh
        prep_info = run_prep_sbas(
            project_root, orbit, subswath, unwrap_file, corr_file
        )
        scene_tab = sbas_dir / "scene.tab"
        intf_tab = sbas_dir / "intf.tab"
        num_scenes = prep_info.get("num_scenes", 0)
        num_intfs = prep_info.get("num_interferograms", 0)
    else:
        # Manual preparation
        scene_tab, num_scenes = prepare_scene_tab(baseline_table, sbas_dir)
        intf_tab, num_intfs = prepare_intf_tab(
            intf_in, baseline_table, merge_dir, sbas_dir, unwrap_file, corr_file
        )
        prep_info = {
            "scene_tab": str(scene_tab),
            "intf_tab": str(intf_tab),
            "num_scenes": num_scenes,
            "num_interferograms": num_intfs,
        }

    logger.info(f"  Prepared scene.tab with {num_scenes} scenes")
    logger.info(f"  Prepared intf.tab with {num_intfs} interferograms")
    result_msg += f"Prepared SBAS inputs: {num_scenes} scenes, {num_intfs} interferograms\n"

    if num_intfs == 0:
        raise RuntimeError("No valid interferograms found for SBAS")

    # =================================================================
    # STEP 2: Get grid dimensions and radar parameters (gmt grdinfo)
    # =================================================================
    logger.info("-" * 50)
    logger.info("STEP 2: Extracting grid dimensions and radar parameters (gmt grdinfo)")
    logger.info("-" * 50)

    # Find a sample unwrap.grd to get dimensions
    sample_intf = None
    for d in merge_dir.iterdir():
        if d.is_dir() and "_" in d.name:
            sample_unwrap = d / unwrap_file
            if sample_unwrap.exists():
                sample_intf = sample_unwrap
                break

    if sample_intf is None:
        raise RuntimeError(f"No {unwrap_file} found in merge directory")

    xdim, ydim = get_grid_dimensions(sample_intf)
    logger.info(f"  Grid dimensions: {xdim} x {ydim}")

    # Get x_min, x_max for range calculation
    cmd = f"gmt grdinfo -C {sample_intf}"
    rc, out, _ = run_cmd(cmd)
    parts = out.strip().split()
    x_min, x_max = float(parts[1]), float(parts[2])

    # Find master PRM file
    raw_dir = project_root / orbit / subswath / "raw"
    prm_files = list(raw_dir.glob("*.PRM"))
    if not prm_files:
        raise RuntimeError(f"No PRM files found in {raw_dir}")

    master_prm = prm_files[0]
    params = get_radar_parameters(master_prm)
    wavelength = params.get("wavelength", 0.0554658)
    range_dist = calculate_range_distance(master_prm, x_min, x_max)

    logger.info(f"  Wavelength: {wavelength} m")
    logger.info(f"  Range distance: {range_dist:.0f} m")

    prep_info["grid_dimensions"] = {"xdim": xdim, "ydim": ydim}
    prep_info["radar_parameters"] = {
        "wavelength": wavelength,
        "range": range_dist,
    }

    # =================================================================
    # STEP 3: Run SBAS inversion (sbas command)
    # =================================================================
    logger.info("-" * 50)
    logger.info(f"STEP 3: Running SBAS inversion (smooth={smooth}, atm_iterations={atm_iterations}, mmap={use_mmap})")
    logger.info("-" * 50)

    sbas_info = run_sbas_inversion(
        sbas_dir=sbas_dir,
        intf_tab=intf_tab,
        scene_tab=scene_tab,
        num_intfs=num_intfs,
        num_scenes=num_scenes,
        xdim=xdim,
        ydim=ydim,
        wavelength=wavelength,
        incidence=37.0,  # Default for Sentinel-1
        range_dist=range_dist,
        smooth=smooth,
        compute_rms=True,
        compute_dem_err=True,
        atm_iterations=atm_iterations,
        use_mmap=use_mmap
    )

    if sbas_info["return_code"] != 0:
        result_msg += f"SBAS inversion FAILED with return code {sbas_info['return_code']}\n"
        logger.error(f"SBAS failed with return code {sbas_info['return_code']}. Check logs.")
    else:
        result_msg += f"SBAS inversion completed successfully\n"
        logger.info("  SBAS inversion completed successfully")

        # List outputs
        for name, info in sbas_info.get("outputs", {}).items():
            if info.get("exists"):
                logger.info(f"  Created: {name} - {info.get('description', '')}")

        disp_count = sbas_info.get("displacement_grids", {}).get("count", 0)
        logger.info(f"  Created: {disp_count} displacement grids (disp_*.grd)")
        result_msg += f"Created {disp_count} displacement time series grids\n"

    # =================================================================
    # STEP 4: Generate visualizations (matplotlib)
    # =================================================================
    logger.info("-" * 50)
    logger.info("STEP 4: Generating visualizations (matplotlib)")
    logger.info("-" * 50)

    viz_dir = sbas_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    viz_info = {"output_dir": str(viz_dir), "plots": {}}

    # Baseline network plot
    network_plot = viz_dir / "baseline_network.png"
    if create_baseline_network_plot(intf_tab, scene_tab, network_plot):
        viz_info["plots"]["baseline_network"] = str(network_plot)
        result_msg += f"Created baseline network plot\n"

    # Connectivity histogram
    conn_plot = viz_dir / "connectivity_histogram.png"
    if create_connectivity_histogram(intf_tab, scene_tab, conn_plot):
        viz_info["plots"]["connectivity"] = str(conn_plot)
        result_msg += f"Created connectivity histogram\n"

    # Temporal baseline histogram
    temp_plot = viz_dir / "baseline_histograms.png"
    if create_temporal_baseline_histogram(intf_tab, scene_tab, temp_plot):
        viz_info["plots"]["baseline_histograms"] = str(temp_plot)
        result_msg += f"Created baseline histograms\n"

    # =================================================================
    # STEP 5: Write metadata log
    # =================================================================
    logger.info("-" * 50)
    logger.info("STEP 5: Writing metadata log")
    logger.info("-" * 50)

    log_path = write_meta_log(project_root, orbit, prep_info, sbas_info, viz_info)
    logger.info(f"  Metadata saved to: {log_path}")
    result_msg += f"Metadata saved to {log_path}\n"

    logger.info("=" * 70)
    logger.info("STAGE 09 COMPLETED")
    logger.info("=" * 70)

    return log_path, result_msg


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Command-line entry point for Stage 09 SBAS."""
    parser = argparse.ArgumentParser(
        description="GMTSAR Stage 09: SBAS (Small BAseline Subset) Analysis - DRAFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SBAS with default settings
  python run_sbas_09_DRAFT.py /path/to/project asc

  # Run with custom smoothing
  python run_sbas_09_DRAFT.py /path/to/project asc --smooth 3.0

  # Use GNSS-corrected interferograms
  python run_sbas_09_DRAFT.py /path/to/project asc --unwrap-file gnss_corrected_intf.grd

NOTE: This is a DRAFT module - not integrated into main.py
        """
    )

    parser.add_argument(
        "project_root",
        help="Path to project root directory"
    )
    parser.add_argument(
        "orbit",
        help="Orbit directory (asc/des)"
    )
    parser.add_argument(
        "--subswath", "-s",
        default="F1",
        help="Subswath for baseline_table.dat (default: F1)"
    )
    parser.add_argument(
        "--unwrap-file", "-u",
        default="unwrap.grd",
        help="Name of unwrapped phase file (default: unwrap.grd)"
    )
    parser.add_argument(
        "--corr-file", "-c",
        default="corr.grd",
        help="Name of correlation file (default: corr.grd)"
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=5.0,
        help="Smoothing factor for SBAS (default: 5.0)"
    )
    parser.add_argument(
        "--atm-iterations",
        type=int,
        default=0,
        help="Number of atmospheric iterations (default: 0 = none)"
    )
    parser.add_argument(
        "--manual-prep",
        action="store_true",
        help="Use manual preparation instead of prep_sbas.csh"
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    print(f"Project root: {project_root}")
    print(f"Orbit: {args.orbit}")
    print(f"Subswath: {args.subswath}")
    print(f"Unwrap file: {args.unwrap_file}")
    print(f"Smooth factor: {args.smooth}")
    print("-" * 60)

    log_path, result_msg = run_sbas(
        project_root=project_root,
        orbit=args.orbit,
        subswath=args.subswath,
        unwrap_file=args.unwrap_file,
        corr_file=args.corr_file,
        smooth=args.smooth,
        atm_iterations=args.atm_iterations,
        use_prep_script=not args.manual_prep
    )

    print(result_msg)
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
