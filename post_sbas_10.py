#!/usr/bin/env python3
"""
================================================================================
GMTSAR Stage 10: Post-SBAS Processing and Visualization================================================================================

This module implements post-SBAS processing following the GMTSAR Sentinel-1
tutorial (Section 10c).

Post-SBAS processing includes:
1. Project SBAS results from radar to lat/lon coordinates (proj_ra2ll.csh)
2. Create color palettes and KML files for Google Earth visualization
3. Project displacement time series grids
4. Generate comprehensive visualizations (velocity maps, time series plots, etc.)

WORKFLOW (from PDF manual Section 10c):
1. Link trans.dat and gauss filter file to SBAS directory
2. Project vel.grd to vel_ll.grd using proj_ra2ll.csh
3. Create color palette with gmt grd2cpt
4. Generate KML overlay with grd2kml.csh
5. Optionally project displacement grids and create animations

STATUS: Production - integrated into main.py
Author: Claude Code Assistant
Date: 2026-02-11
================================================================================
"""

import argparse
import subprocess
import json
import os
from pathlib import Path
import datetime
from typing import Dict, Any, Optional, List, Tuple

# Optional imports for visualization
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_cmd(cmd: str, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """
    Execute a shell command and return (returncode, stdout, stderr).
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


def read_gmt_grid(grd_file: Path) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Read a GMT grid file using GMT's grd2xyz or netCDF4.

    Returns:
        (data_array, metadata_dict) or None if failed
    """
    if not HAS_NUMPY:
        return None

    try:
        # Try using netCDF4 first (faster)
        try:
            from netCDF4 import Dataset
            ds = Dataset(grd_file, 'r')
            z = ds.variables['z'][:]
            x = ds.variables['x'][:]
            y = ds.variables['y'][:]
            ds.close()

            metadata = {
                'x_min': float(x.min()),
                'x_max': float(x.max()),
                'y_min': float(y.min()),
                'y_max': float(y.max()),
                'nx': len(x),
                'ny': len(y),
            }
            return np.array(z), metadata

        except ImportError:
            # Fallback to gmt grd2xyz
            cmd = f"gmt grd2xyz {grd_file}"
            rc, out, err = run_cmd(cmd)
            if rc != 0:
                return None

            lines = out.strip().split('\n')
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    data.append([float(p) for p in parts[:3]])

            if not data:
                return None

            data = np.array(data)
            # Reshape based on unique x,y values
            x_unique = np.unique(data[:, 0])
            y_unique = np.unique(data[:, 1])
            nx, ny = len(x_unique), len(y_unique)

            z = data[:, 2].reshape((ny, nx))

            metadata = {
                'x_min': x_unique.min(),
                'x_max': x_unique.max(),
                'y_min': y_unique.min(),
                'y_max': y_unique.max(),
                'nx': nx,
                'ny': ny,
            }
            return z, metadata

    except Exception as e:
        print(f"WARNING: Could not read grid {grd_file}: {e}")
        return None


def get_grid_stats(grd_file: Path) -> Dict[str, float]:
    """
    Get statistics of a GMT grid file using gmt grdinfo.

    Returns:
        Dict with min, max, mean, std, etc.
    """
    cmd = f"gmt grdinfo -C {grd_file}"
    rc, out, err = run_cmd(cmd)

    if rc != 0:
        return {}

    # Format: file xmin xmax ymin ymax zmin zmax dx dy nx ny [registration]
    parts = out.strip().split()
    if len(parts) >= 11:
        return {
            'x_min': float(parts[1]),
            'x_max': float(parts[2]),
            'y_min': float(parts[3]),
            'y_max': float(parts[4]),
            'z_min': float(parts[5]),
            'z_max': float(parts[6]),
            'dx': float(parts[7]),
            'dy': float(parts[8]),
            'nx': int(parts[9]),
            'ny': int(parts[10]),
        }

    return {}


# =============================================================================
# PROJECTION FUNCTIONS
# =============================================================================

def project_to_latlon(
    sbas_dir: Path,
    input_grd: str,
    output_grd: str,
    trans_dat: Path
) -> Dict[str, Any]:
    """
    Project a grid from radar coordinates to lat/lon using proj_ra2ll.csh.

    This creates raln.grd and ralt.grd lookup tables on first run.

    Args:
        sbas_dir: SBAS working directory
        input_grd: Input grid filename (in radar coordinates)
        output_grd: Output grid filename (will be in lat/lon)
        trans_dat: Path to trans.dat file

    Returns:
        Dict with projection results
    """
    # Link trans.dat if not already present
    trans_link = sbas_dir / "trans.dat"
    if not trans_link.exists():
        os.symlink(trans_dat, trans_link)

    cmd = f"proj_ra2ll.csh trans.dat {input_grd} {output_grd}"
    rc, out, err = run_cmd(cmd, cwd=sbas_dir)

    result = {
        "command": cmd,
        "return_code": rc,
        "input": input_grd,
        "output": output_grd,
        "stdout": out[-1000:] if len(out) > 1000 else out,
        "stderr": err[-1000:] if len(err) > 1000 else err,
    }

    output_path = sbas_dir / output_grd
    if output_path.exists():
        result["success"] = True
        result["output_path"] = str(output_path)
        stats = get_grid_stats(output_path)
        result["stats"] = stats
    else:
        result["success"] = False

    return result


def create_color_palette(
    sbas_dir: Path,
    input_grd: str,
    output_cpt: str,
    cmap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    reverse: bool = False
) -> Dict[str, Any]:
    """
    Create a GMT color palette for a grid file.

    Args:
        sbas_dir: SBAS working directory
        input_grd: Input grid filename
        output_cpt: Output CPT filename
        cmap: GMT colormap name (default: jet)
        vmin: Minimum value (None = auto)
        vmax: Maximum value (None = auto)
        reverse: Reverse the colormap

    Returns:
        Dict with creation results
    """
    grd_path = sbas_dir / input_grd

    # Get grid statistics if vmin/vmax not specified
    if vmin is None or vmax is None:
        stats = get_grid_stats(grd_path)
        if vmin is None:
            vmin = stats.get('z_min', -100)
        if vmax is None:
            vmax = stats.get('z_max', 100)

    # Calculate step size
    step = (vmax - vmin) / 50

    # Build command
    cmd = f"gmt grd2cpt {input_grd} -Z -C{cmap} -T{vmin}/{vmax}/{step}"
    if reverse:
        cmd += " -I"
    cmd += f" > {output_cpt}"

    rc, out, err = run_cmd(cmd, cwd=sbas_dir)

    result = {
        "command": cmd,
        "return_code": rc,
        "output_cpt": output_cpt,
        "vmin": vmin,
        "vmax": vmax,
        "colormap": cmap,
    }

    cpt_path = sbas_dir / output_cpt
    result["success"] = cpt_path.exists()

    return result


def create_kml_overlay(
    sbas_dir: Path,
    input_grd: str,
    cpt_file: str,
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a KML file with PNG overlay for Google Earth visualization.

    Uses GMTSAR's grd2kml.csh script.

    Args:
        sbas_dir: SBAS working directory
        input_grd: Input grid filename (must be in lat/lon)
        cpt_file: Color palette file
        output_name: Base name for output (default: same as input without .grd)

    Returns:
        Dict with creation results
    """
    if output_name is None:
        output_name = input_grd.replace(".grd", "")

    cmd = f"grd2kml.csh {output_name} {cpt_file}"
    rc, out, err = run_cmd(cmd, cwd=sbas_dir)

    result = {
        "command": cmd,
        "return_code": rc,
        "input_grid": input_grd,
        "color_palette": cpt_file,
    }

    kml_path = sbas_dir / f"{output_name}.kml"
    png_path = sbas_dir / f"{output_name}.png"

    result["kml_file"] = str(kml_path) if kml_path.exists() else None
    result["png_file"] = str(png_path) if png_path.exists() else None
    result["success"] = kml_path.exists() and png_path.exists()

    return result


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_velocity_map(
    sbas_dir: Path,
    vel_grd: str,
    output_path: Path,
    title: str = "LOS Velocity",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdYlBu_r"
) -> bool:
    """
    Create a publication-quality velocity map.

    Args:
        sbas_dir: SBAS directory
        vel_grd: Velocity grid filename
        output_path: Output image path
        title: Plot title
        vmin: Minimum velocity for colorbar
        vmax: Maximum velocity for colorbar
        cmap: Matplotlib colormap name

    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("WARNING: matplotlib/numpy not available, skipping velocity map")
        return False

    try:
        grd_path = sbas_dir / vel_grd
        result = read_gmt_grid(grd_path)

        if result is None:
            print(f"WARNING: Could not read {grd_path}")
            return False

        data, meta = result

        # Handle NaN values
        data = np.ma.masked_invalid(data)

        # Auto-scale if not provided
        if vmin is None:
            vmin = np.nanpercentile(data.compressed(), 2)
        if vmax is None:
            vmax = np.nanpercentile(data.compressed(), 98)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot data
        extent = [meta['x_min'], meta['x_max'], meta['y_min'], meta['y_max']]
        im = ax.imshow(
            data,
            extent=extent,
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('LOS Velocity (mm/yr)', fontsize=12)

        # Labels
        ax.set_xlabel('Range', fontsize=12)
        ax.set_ylabel('Azimuth', fontsize=12)
        ax.set_title(f'{title}\nRange: [{vmin:.1f}, {vmax:.1f}] mm/yr', fontsize=14)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved velocity map to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR creating velocity map: {e}")
        return False


def create_rms_map(
    sbas_dir: Path,
    rms_grd: str,
    output_path: Path
) -> bool:
    """
    Create an RMS residual map showing SBAS fit quality.

    Lower RMS values indicate better fit.

    Args:
        sbas_dir: SBAS directory
        rms_grd: RMS grid filename
        output_path: Output image path

    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("WARNING: matplotlib/numpy not available, skipping RMS map")
        return False

    try:
        grd_path = sbas_dir / rms_grd
        result = read_gmt_grid(grd_path)

        if result is None:
            print(f"WARNING: Could not read {grd_path}")
            return False

        data, meta = result
        data = np.ma.masked_invalid(data)

        # RMS is always positive, use appropriate colormap
        vmin = 0
        vmax = np.nanpercentile(data.compressed(), 95)

        fig, ax = plt.subplots(figsize=(12, 10))

        extent = [meta['x_min'], meta['x_max'], meta['y_min'], meta['y_max']]
        im = ax.imshow(
            data,
            extent=extent,
            origin='lower',
            cmap='YlOrRd',
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('RMS Residual (mm)', fontsize=12)

        ax.set_xlabel('Range', fontsize=12)
        ax.set_ylabel('Azimuth', fontsize=12)
        ax.set_title(f'SBAS RMS Residuals\nMax shown: {vmax:.1f} mm', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved RMS map to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR creating RMS map: {e}")
        return False


def create_dem_error_map(
    sbas_dir: Path,
    dem_err_grd: str,
    output_path: Path
) -> bool:
    """
    Create a DEM error map showing topographic phase residuals.

    Args:
        sbas_dir: SBAS directory
        dem_err_grd: DEM error grid filename
        output_path: Output image path

    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("WARNING: matplotlib/numpy not available, skipping DEM error map")
        return False

    try:
        grd_path = sbas_dir / dem_err_grd
        result = read_gmt_grid(grd_path)

        if result is None:
            print(f"WARNING: Could not read {grd_path}")
            return False

        data, meta = result
        data = np.ma.masked_invalid(data)

        # DEM error can be positive or negative
        vabs = np.nanpercentile(np.abs(data.compressed()), 95)
        vmin, vmax = -vabs, vabs

        fig, ax = plt.subplots(figsize=(12, 10))

        extent = [meta['x_min'], meta['x_max'], meta['y_min'], meta['y_max']]
        im = ax.imshow(
            data,
            extent=extent,
            origin='lower',
            cmap='RdBu',
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('DEM Error (m)', fontsize=12)

        ax.set_xlabel('Range', fontsize=12)
        ax.set_ylabel('Azimuth', fontsize=12)
        ax.set_title(f'DEM Error Estimate\nRange: [{vmin:.1f}, {vmax:.1f}] m', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved DEM error map to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR creating DEM error map: {e}")
        return False


def create_displacement_timeseries_plot(
    sbas_dir: Path,
    scene_tab: Path,
    sample_points: List[Tuple[int, int]],
    output_path: Path,
    point_labels: Optional[List[str]] = None
) -> bool:
    """
    Create time series plot of displacement at selected points.

    Args:
        sbas_dir: SBAS directory
        scene_tab: Path to scene.tab file
        sample_points: List of (x, y) pixel coordinates to extract
        output_path: Output image path
        point_labels: Optional labels for each point

    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("WARNING: matplotlib/numpy not available, skipping time series plot")
        return False

    try:
        # Parse scene.tab for dates
        dates = []
        days = []
        with open(scene_tab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    scene_id = parts[0]
                    day = int(parts[1])
                    days.append(day)
                    # Convert to date (reference: 2014-01-01 for Sentinel-1)
                    date = datetime.date(2014, 1, 1) + datetime.timedelta(days=day)
                    dates.append(date)

        # Find displacement grids
        disp_grids = sorted(sbas_dir.glob("disp_*.grd"))
        if not disp_grids:
            print("WARNING: No displacement grids found")
            return False

        # Extract time series at each point
        time_series = {i: [] for i in range(len(sample_points))}

        for grd_file in disp_grids:
            result = read_gmt_grid(grd_file)
            if result is None:
                continue

            data, meta = result

            for i, (px, py) in enumerate(sample_points):
                # Bounds checking
                if 0 <= py < data.shape[0] and 0 <= px < data.shape[1]:
                    val = data[py, px]
                    if not np.isnan(val):
                        time_series[i].append(val)
                    else:
                        time_series[i].append(np.nan)
                else:
                    time_series[i].append(np.nan)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_points)))

        for i, (px, py) in enumerate(sample_points):
            ts = time_series[i]
            if len(ts) == len(dates):
                label = point_labels[i] if point_labels and i < len(point_labels) else f"Point ({px}, {py})"
                ax.plot(dates[:len(ts)], ts, 'o-', color=colors[i], label=label, markersize=4)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('LOS Displacement (mm)', fontsize=12)
        ax.set_title('Displacement Time Series at Selected Points', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved time series plot to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR creating time series plot: {e}")
        return False


def extract_gnss_point_timeseries(
    sbas_dir: Path,
    scene_tab: Path,
    gnss_file: Path,
    output_dir: Path
) -> bool:
    """
    Extract displacement time series at GNSS lon/lat points using gmt grdtrack
    on projected disp_*_ll.grd files, and create individual per-point plots
    with linear trend lines.

    Args:
        sbas_dir: SBAS directory containing disp_*_ll.grd and vel_ll.grd
        scene_tab: Path to scene.tab file (for dates)
        gnss_file: Path to GNSS.ll file with "lon lat" per line
        output_dir: Directory to write point_*.png plots

    Returns:
        True if at least one plot was created
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("WARNING: matplotlib/numpy not available, skipping GNSS point plots")
        return False

    try:
        # 1. Parse GNSS.ll file
        points = []
        with open(gnss_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    points.append((lon, lat))

        if not points:
            print("WARNING: No valid points found in GNSS.ll file")
            return False

        print(f"Read {len(points)} GNSS points from {gnss_file}")

        # 2. Parse scene.tab for dates
        dates = []
        with open(scene_tab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    day = int(parts[1])
                    date = datetime.date(2014, 1, 1) + datetime.timedelta(days=day)
                    dates.append(date)

        # 3. Find projected displacement grids
        disp_ll_grids = sorted(sbas_dir.glob("disp_*_ll.grd"))
        if not disp_ll_grids:
            print("WARNING: No projected displacement grids (disp_*_ll.grd) found")
            return False

        print(f"Found {len(disp_ll_grids)} projected displacement grids")

        # 4. Create a temporary file with GNSS points for gmt grdtrack
        import tempfile
        gnss_tmp = sbas_dir / "gnss_points.tmp"
        with open(gnss_tmp, 'w') as f:
            for lon, lat in points:
                f.write(f"{lon} {lat}\n")

        # 5. Extract displacement time series at each point
        # time_series[point_idx] = list of (date, displacement_value)
        time_series = {i: [] for i in range(len(points))}

        for k, grd_file in enumerate(disp_ll_grids):
            # Use gmt grdtrack to sample this grid at GNSS points
            cmd = f"gmt grdtrack {gnss_tmp} -G{grd_file}"
            rc, stdout, stderr = run_cmd(cmd, cwd=sbas_dir)

            if rc != 0:
                continue

            # Parse output: each line is "lon lat value"
            lines = stdout.strip().split('\n')
            date_idx = k if k < len(dates) else None

            for i, line in enumerate(lines):
                if i >= len(points):
                    break
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        val = float(parts[2])
                    except ValueError:
                        val = float('nan')
                else:
                    val = float('nan')

                if date_idx is not None and date_idx < len(dates):
                    time_series[i].append((dates[date_idx], val))

        # 6. Extract velocity at each point (for reference in title)
        vel_ll = sbas_dir / "vel_ll.grd"
        point_velocities = [None] * len(points)
        if vel_ll.exists():
            cmd = f"gmt grdtrack {gnss_tmp} -G{vel_ll}"
            rc, stdout, stderr = run_cmd(cmd, cwd=sbas_dir)
            if rc == 0:
                lines = stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if i >= len(points):
                        break
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            point_velocities[i] = float(parts[2])
                        except ValueError:
                            pass

        # 7. Clean up temp file
        if gnss_tmp.exists():
            gnss_tmp.unlink()

        # 8. Create individual plots
        import matplotlib.dates as mdates
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_created = 0

        for i, (lon, lat) in enumerate(points):
            ts = time_series[i]
            if not ts:
                continue

            # Filter out NaN values
            valid_ts = [(d, v) for d, v in ts if not np.isnan(v)]
            if len(valid_ts) < 2:
                continue

            ts_dates, ts_vals = zip(*valid_ts)
            ts_dates = list(ts_dates)
            ts_vals = list(ts_vals)

            fig, ax = plt.subplots(figsize=(14, 6))

            # Scatter plot
            ax.plot(ts_dates, ts_vals, 'o', color='blue', markersize=5, zorder=3)

            # Linear trend line
            # Convert dates to decimal years for fitting
            date_nums = np.array([(d - ts_dates[0]).days / 365.25 for d in ts_dates])
            vals_arr = np.array(ts_vals)
            coeffs = np.polyfit(date_nums, vals_arr, 1)
            trend_rate = coeffs[0]  # mm/yr

            # Draw trend line
            trend_line = np.polyval(coeffs, date_nums)
            ax.plot(ts_dates, trend_line, '--', color='red', linewidth=1.5,
                    label=f'Trend: {trend_rate:.1f} mm/yr', zorder=2)

            # Zero line
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, zorder=1)

            # Labels and title
            point_num = i + 1
            title = f"Displacement Time Series - Point {point_num}\n"
            title += f"Location: ({lon}\u00b0E, {lat}\u00b0N)"
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('LOS Displacement (mm)', fontsize=12)
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())

            plt.tight_layout()
            plot_path = output_dir / f"point_{point_num}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plots_created += 1

        print(f"Created {plots_created} GNSS point time series plots in {output_dir}")
        return plots_created > 0

    except Exception as e:
        print(f"ERROR creating GNSS point time series: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_velocity_statistics_report(
    sbas_dir: Path,
    output_path: Path,
    orbit: str = "asc"
) -> bool:
    """
    Generate a formatted text report of vel_ll.grd statistics.

    Args:
        sbas_dir: SBAS directory containing vel_ll.grd
        output_path: Output text file path
        orbit: Orbit name for labeling ("asc" or "des")

    Returns:
        True if report was created successfully
    """
    vel_ll = sbas_dir / "vel_ll.grd"
    if not vel_ll.exists():
        print("WARNING: vel_ll.grd not found, skipping statistics report")
        return False

    try:
        # Get grid info via gmt grdinfo
        stats = get_grid_stats(vel_ll)
        if not stats:
            print("WARNING: Could not read vel_ll.grd statistics")
            return False

        nx = int(stats.get('nx', 0))
        ny = int(stats.get('ny', 0))
        x_min = stats.get('x_min', 0)
        x_max = stats.get('x_max', 0)
        y_min = stats.get('y_min', 0)
        y_max = stats.get('y_max', 0)
        z_min = stats.get('z_min', 0)
        z_max = stats.get('z_max', 0)
        dx = stats.get('dx', 0)
        dy = stats.get('dy', 0)

        total_pixels = nx * ny

        # Try to get full statistics using read_gmt_grid
        mean_val = median_val = std_val = rms_val = None
        valid_count = nan_count = None

        if HAS_NUMPY:
            result = read_gmt_grid(vel_ll)
            if result is not None:
                data, meta = result
                valid_mask = ~np.isnan(data)
                valid_data = data[valid_mask]
                valid_count = int(valid_mask.sum())
                nan_count = int((~valid_mask).sum())
                if len(valid_data) > 0:
                    mean_val = float(np.mean(valid_data))
                    median_val = float(np.median(valid_data))
                    std_val = float(np.std(valid_data))
                    rms_val = float(np.sqrt(np.mean(valid_data ** 2)))

        # Build report
        orbit_label = "Ascending" if orbit.lower().startswith("asc") else "Descending"
        today = datetime.date.today().strftime("%Y-%m-%d")

        lines = []
        lines.append("=" * 80)
        lines.append("                    SBAS-InSAR Velocity Statistics Report")
        lines.append("                         vel_ll.grd Analysis")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Date: {today}")
        lines.append(f"Directory: {sbas_dir}")
        lines.append(f"Orbit: {orbit_label}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("                              GRID INFORMATION")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"File:               vel_ll.grd")
        lines.append(f"Dimensions:         {nx} x {ny} pixels")
        lines.append("")
        lines.append("Geographic Extent:")
        lines.append(f"  Longitude:        {x_min:.4f}\u00b0 E  to  {x_max:.4f}\u00b0 E")
        lines.append(f"  Latitude:         {y_min:.4f}\u00b0 N  to  {y_max:.4f}\u00b0 N")
        if dx and dy:
            lines.append(f"  Resolution:       {dx} (lon) x {dy} (lat)")
        lines.append("")
        lines.append("-" * 80)
        lines.append("                           BASIC STATISTICS")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"Total pixels:        {total_pixels:>12,}")
        if valid_count is not None:
            valid_pct = 100.0 * valid_count / total_pixels if total_pixels > 0 else 0
            nan_pct = 100.0 * nan_count / total_pixels if total_pixels > 0 else 0
            lines.append(f"Valid pixels:        {valid_count:>12,}  ({valid_pct:.1f}%)")
            lines.append(f"NaN pixels:          {nan_count:>12,}  ({nan_pct:.1f}%)")
        lines.append("")
        lines.append(f"Minimum:            {z_min:+.2f} mm/year")
        lines.append(f"Maximum:            {z_max:+.2f} mm/year")
        if mean_val is not None:
            lines.append(f"Mean:               {mean_val:+.2f} mm/year")
        if median_val is not None:
            lines.append(f"Median:             {median_val:+.2f} mm/year")
        if std_val is not None:
            lines.append(f"Std Deviation:      {std_val:+.2f} mm/year")
        if rms_val is not None:
            lines.append(f"RMS:                {rms_val:+.2f} mm/year")
        lines.append("")
        lines.append("=" * 80)

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        print(f"Saved velocity statistics report to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR creating velocity statistics report: {e}")
        return False


def create_velocity_histogram(
    sbas_dir: Path,
    vel_grd: str,
    output_path: Path
) -> bool:
    """
    Create histogram of velocity values.

    Args:
        sbas_dir: SBAS directory
        vel_grd: Velocity grid filename
        output_path: Output image path

    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("WARNING: matplotlib/numpy not available, skipping velocity histogram")
        return False

    try:
        grd_path = sbas_dir / vel_grd
        result = read_gmt_grid(grd_path)

        if result is None:
            return False

        data, meta = result
        data = data.flatten()
        data = data[~np.isnan(data)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Full histogram
        ax1.hist(data, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.1f}')
        ax1.axvline(x=np.median(data), color='orange', linestyle=':', label=f'Median: {np.median(data):.1f}')
        ax1.set_xlabel('LOS Velocity (mm/yr)', fontsize=12)
        ax1.set_ylabel('Pixel Count', fontsize=12)
        ax1.set_title('Velocity Distribution (Full Range)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Clipped histogram (central 98%)
        vmin, vmax = np.percentile(data, [1, 99])
        data_clipped = data[(data >= vmin) & (data <= vmax)]
        ax2.hist(data_clipped, bins=100, color='coral', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(data_clipped), color='red', linestyle='--')
        ax2.set_xlabel('LOS Velocity (mm/yr)', fontsize=12)
        ax2.set_ylabel('Pixel Count', fontsize=12)
        ax2.set_title(f'Velocity Distribution (1-99 percentile)\nStd: {np.std(data_clipped):.1f} mm/yr', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved velocity histogram to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR creating velocity histogram: {e}")
        return False


def create_summary_figure(
    sbas_dir: Path,
    output_path: Path
) -> bool:
    """
    Create a multi-panel summary figure with all SBAS results.

    Panels:
    - Velocity map
    - RMS map
    - DEM error map
    - Velocity histogram

    Args:
        sbas_dir: SBAS directory
        output_path: Output image path

    Returns:
        True if successful
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("WARNING: matplotlib/numpy not available, skipping summary figure")
        return False

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Panel 1: Velocity map
        vel_grd = sbas_dir / "vel.grd"
        if vel_grd.exists():
            result = read_gmt_grid(vel_grd)
            if result:
                data, meta = result
                data = np.ma.masked_invalid(data)
                vabs = np.nanpercentile(np.abs(data.compressed()), 98)
                extent = [meta['x_min'], meta['x_max'], meta['y_min'], meta['y_max']]
                im = axes[0, 0].imshow(data, extent=extent, origin='lower',
                                       cmap='RdYlBu_r', vmin=-vabs, vmax=vabs, aspect='auto')
                plt.colorbar(im, ax=axes[0, 0], shrink=0.8, label='mm/yr')
                axes[0, 0].set_title('LOS Velocity')
        else:
            axes[0, 0].text(0.5, 0.5, 'vel.grd not found', ha='center', va='center')
            axes[0, 0].set_title('LOS Velocity')

        # Panel 2: RMS map
        rms_grd = sbas_dir / "rms.grd"
        if rms_grd.exists():
            result = read_gmt_grid(rms_grd)
            if result:
                data, meta = result
                data = np.ma.masked_invalid(data)
                vmax = np.nanpercentile(data.compressed(), 95)
                extent = [meta['x_min'], meta['x_max'], meta['y_min'], meta['y_max']]
                im = axes[0, 1].imshow(data, extent=extent, origin='lower',
                                       cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
                plt.colorbar(im, ax=axes[0, 1], shrink=0.8, label='mm')
                axes[0, 1].set_title('RMS Residuals')
        else:
            axes[0, 1].text(0.5, 0.5, 'rms.grd not found', ha='center', va='center')
            axes[0, 1].set_title('RMS Residuals')

        # Panel 3: DEM error map
        dem_grd = sbas_dir / "dem_err.grd"
        if dem_grd.exists():
            result = read_gmt_grid(dem_grd)
            if result:
                data, meta = result
                data = np.ma.masked_invalid(data)
                vabs = np.nanpercentile(np.abs(data.compressed()), 95)
                extent = [meta['x_min'], meta['x_max'], meta['y_min'], meta['y_max']]
                im = axes[1, 0].imshow(data, extent=extent, origin='lower',
                                       cmap='RdBu', vmin=-vabs, vmax=vabs, aspect='auto')
                plt.colorbar(im, ax=axes[1, 0], shrink=0.8, label='m')
                axes[1, 0].set_title('DEM Error')
        else:
            axes[1, 0].text(0.5, 0.5, 'dem_err.grd not found', ha='center', va='center')
            axes[1, 0].set_title('DEM Error')

        # Panel 4: Velocity histogram
        if vel_grd.exists():
            result = read_gmt_grid(vel_grd)
            if result:
                data, _ = result
                data = data.flatten()
                data = data[~np.isnan(data)]
                vmin, vmax = np.percentile(data, [1, 99])
                data_clipped = data[(data >= vmin) & (data <= vmax)]
                axes[1, 1].hist(data_clipped, bins=50, color='steelblue', alpha=0.7)
                axes[1, 1].axvline(x=np.mean(data_clipped), color='red', linestyle='--',
                                   label=f'Mean: {np.mean(data_clipped):.1f}')
                axes[1, 1].legend()
                axes[1, 1].set_xlabel('LOS Velocity (mm/yr)')
                axes[1, 1].set_ylabel('Pixel Count')
                axes[1, 1].set_title('Velocity Distribution')
                axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No velocity data', ha='center', va='center')
            axes[1, 1].set_title('Velocity Distribution')

        plt.suptitle('SBAS Analysis Summary', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Saved summary figure to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR creating summary figure: {e}")
        return False


# =============================================================================
# METADATA LOGGING
# =============================================================================

def write_meta_log(
    project_root: Path,
    orbit: str,
    projection_info: Dict,
    kml_info: Dict,
    viz_info: Dict
) -> Path:
    """
    Write metadata log for Stage 10.

    Args:
        project_root: Project root directory
        orbit: Orbit directory name
        projection_info: Projection results
        kml_info: KML creation results
        viz_info: Visualization results

    Returns:
        Path to log file
    """
    meta = {
        "step": 10,
        "stage_name": "Post-SBAS Processing",
        "orbit": orbit,
        "timestamp": datetime.datetime.now().isoformat(),
        "projection": projection_info,
        "kml_outputs": kml_info,
        "visualizations": viz_info,
    }

    log_dir = project_root / "wrapper_meta" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"step10_{orbit}_post_sbas.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return log_path


# =============================================================================
# MAIN POST-SBAS FUNCTION
# =============================================================================

def run_post_sbas(
    project_root: Path,
    orbit: str = "asc",
    subswath: str = "F2",
    vel_cmap: str = "jet",
    vel_range: Optional[Tuple[float, float]] = None,
    project_disp: bool = True,
    max_disp_grids: int = 50,
    gnss_file: Optional[Path] = None
) -> Tuple[Path, str]:
    """
    Run complete post-SBAS processing pipeline.

    This function:
    1. Projects velocity grid to lat/lon
    2. Creates color palettes and KML files
    3. Optionally projects displacement grids
    4. Generates comprehensive visualizations
    5. Writes metadata log

    Args:
        project_root: Project root directory
        orbit: Orbit directory (asc/des)
        subswath: Subswath to use for trans.dat (default: F2)
        vel_cmap: Colormap for velocity (default: jet)
        vel_range: (vmin, vmax) for velocity colorbar (None = auto)
        project_disp: Whether to project displacement grids
        max_disp_grids: Maximum number of displacement grids to project

    Returns:
        (log_path, result_message)
    """
    print("=" * 70)
    print("STAGE 10: Post-SBAS Processing and Visualization")
    print("=" * 70)

    result_msg = ""

    sbas_dir = project_root / orbit / "SBAS"
    merge_dir = project_root / orbit / "merge"

    if not sbas_dir.exists():
        raise RuntimeError(f"SBAS directory not found: {sbas_dir}")

    # Find trans.dat
    trans_dat = merge_dir / "trans.dat"
    if not trans_dat.exists():
        # Try in F* subswath
        trans_dat = project_root / orbit / subswath / "topo" / "trans.dat"

    if not trans_dat.exists():
        raise RuntimeError(f"trans.dat not found")

    print(f"SBAS directory: {sbas_dir}")
    print(f"Using trans.dat: {trans_dat}")

    # Create visualization directory
    viz_dir = sbas_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    projection_info = {}
    kml_info = {}
    viz_info = {"output_dir": str(viz_dir), "plots": {}}

    # =================================================================
    # STEP 1: Project velocity grid to lat/lon
    # =================================================================
    print("-" * 50)
    print("STEP 1: Projecting velocity grid to lat/lon")
    print("-" * 50)

    vel_grd = sbas_dir / "vel.grd"
    if vel_grd.exists():
        proj_result = project_to_latlon(sbas_dir, "vel.grd", "vel_ll.grd", trans_dat)
        projection_info["vel"] = proj_result

        if proj_result.get("success"):
            print(f"  Projected vel.grd to vel_ll.grd")
            result_msg += "Projected velocity to lat/lon\n"
        else:
            print(f"  WARNING: Velocity projection failed")
    else:
        print("  WARNING: vel.grd not found, skipping")

    # =================================================================
    # STEP 2: Project RMS grid
    # =================================================================
    print("-" * 50)
    print("STEP 2: Projecting RMS grid to lat/lon")
    print("-" * 50)

    rms_grd = sbas_dir / "rms.grd"
    if rms_grd.exists():
        proj_result = project_to_latlon(sbas_dir, "rms.grd", "rms_ll.grd", trans_dat)
        projection_info["rms"] = proj_result

        if proj_result.get("success"):
            print(f"  Projected rms.grd to rms_ll.grd")
            result_msg += "Projected RMS to lat/lon\n"
    else:
        print("  WARNING: rms.grd not found, skipping")

    # =================================================================
    # STEP 3: Project DEM error grid
    # =================================================================
    print("-" * 50)
    print("STEP 3: Projecting DEM error grid to lat/lon")
    print("-" * 50)

    dem_grd = sbas_dir / "dem_err.grd"
    if dem_grd.exists():
        proj_result = project_to_latlon(sbas_dir, "dem_err.grd", "dem_err_ll.grd", trans_dat)
        projection_info["dem_err"] = proj_result

        if proj_result.get("success"):
            print(f"  Projected dem_err.grd to dem_err_ll.grd")
            result_msg += "Projected DEM error to lat/lon\n"
    else:
        print("  WARNING: dem_err.grd not found, skipping")

    # =================================================================
    # STEP 4: Create color palettes and KML files
    # =================================================================
    print("-" * 50)
    print("STEP 4: Creating color palettes and KML overlays")
    print("-" * 50)

    vel_ll = sbas_dir / "vel_ll.grd"
    if vel_ll.exists():
        # Get velocity range
        if vel_range:
            vmin, vmax = vel_range
        else:
            stats = get_grid_stats(vel_ll)
            # Symmetric around zero or use actual range
            vabs = max(abs(stats.get('z_min', -100)), abs(stats.get('z_max', 100)))
            vmin, vmax = -vabs, vabs

        # Create CPT
        cpt_result = create_color_palette(
            sbas_dir, "vel_ll.grd", "vel_ll.cpt",
            cmap=vel_cmap, vmin=vmin, vmax=vmax
        )
        kml_info["vel_cpt"] = cpt_result

        if cpt_result.get("success"):
            print(f"  Created vel_ll.cpt")

            # Create KML
            kml_result = create_kml_overlay(sbas_dir, "vel_ll.grd", "vel_ll.cpt")
            kml_info["vel_kml"] = kml_result

            if kml_result.get("success"):
                print(f"  Created vel_ll.kml and vel_ll.png")
                result_msg += "Created velocity KML overlay for Google Earth\n"

    # =================================================================
    # STEP 5: Project displacement grids (optional)
    # =================================================================
    if project_disp:
        print("-" * 50)
        print("STEP 5: Projecting displacement grids")
        print("-" * 50)

        disp_grids = sorted(sbas_dir.glob("disp_*.grd"))
        disp_grids = [g for g in disp_grids if not g.name.endswith("_ll.grd")]

        if disp_grids:
            num_to_project = min(len(disp_grids), max_disp_grids)
            print(f"  Found {len(disp_grids)} displacement grids, projecting {num_to_project}")

            projection_info["disp_grids"] = {"count": 0, "files": []}

            for i, grd in enumerate(disp_grids[:num_to_project]):
                output_name = grd.stem + "_ll.grd"
                proj_result = project_to_latlon(sbas_dir, grd.name, output_name, trans_dat)

                if proj_result.get("success"):
                    projection_info["disp_grids"]["count"] += 1
                    projection_info["disp_grids"]["files"].append(output_name)

                if (i + 1) % 10 == 0:
                    print(f"    Projected {i + 1}/{num_to_project} grids...")

            print(f"  Projected {projection_info['disp_grids']['count']} displacement grids")
            result_msg += f"Projected {projection_info['disp_grids']['count']} displacement grids\n"
        else:
            print("  No displacement grids found")

    # =================================================================
    # STEP 6: Generate visualizations
    # =================================================================
    print("-" * 50)
    print("STEP 6: Generating visualizations")
    print("-" * 50)

    # Velocity map
    vel_map_path = viz_dir / "velocity_map.png"
    if create_velocity_map(sbas_dir, "vel.grd", vel_map_path):
        viz_info["plots"]["velocity_map"] = str(vel_map_path)
        result_msg += "Created velocity map\n"

    # RMS map
    rms_map_path = viz_dir / "rms_map.png"
    if create_rms_map(sbas_dir, "rms.grd", rms_map_path):
        viz_info["plots"]["rms_map"] = str(rms_map_path)
        result_msg += "Created RMS map\n"

    # DEM error map
    dem_map_path = viz_dir / "dem_error_map.png"
    if create_dem_error_map(sbas_dir, "dem_err.grd", dem_map_path):
        viz_info["plots"]["dem_error_map"] = str(dem_map_path)
        result_msg += "Created DEM error map\n"

    # Velocity histogram
    hist_path = viz_dir / "velocity_histogram.png"
    if create_velocity_histogram(sbas_dir, "vel.grd", hist_path):
        viz_info["plots"]["velocity_histogram"] = str(hist_path)
        result_msg += "Created velocity histogram\n"

    # Summary figure
    summary_path = viz_dir / "sbas_summary.png"
    if create_summary_figure(sbas_dir, summary_path):
        viz_info["plots"]["summary"] = str(summary_path)
        result_msg += "Created summary figure\n"

    # Time series plot (sample points at grid quarters)
    scene_tab = sbas_dir / "scene.tab"
    if scene_tab.exists() and list(sbas_dir.glob("disp_*.grd")):
        # Get grid dimensions for sample points
        sample_grd = list(sbas_dir.glob("disp_*.grd"))[0]
        stats = get_grid_stats(sample_grd)
        nx, ny = stats.get('nx', 100), stats.get('ny', 100)

        sample_points = [
            (nx // 4, ny // 4),
            (3 * nx // 4, ny // 4),
            (nx // 2, ny // 2),
            (nx // 4, 3 * ny // 4),
            (3 * nx // 4, 3 * ny // 4),
        ]
        labels = ["SW", "SE", "Center", "NW", "NE"]

        ts_path = viz_dir / "displacement_timeseries.png"
        if create_displacement_timeseries_plot(sbas_dir, scene_tab, sample_points, ts_path, labels):
            viz_info["plots"]["timeseries"] = str(ts_path)
            result_msg += "Created displacement time series plot\n"

    # =================================================================
    # STEP 6b: GNSS point sampling (if GNSS.ll file provided)
    # =================================================================
    if gnss_file and Path(gnss_file).exists():
        print("-" * 50)
        print("STEP 6b: GNSS point time series extraction")
        print("-" * 50)

        plots_dir = sbas_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        scene_tab = sbas_dir / "scene.tab"

        if scene_tab.exists():
            if extract_gnss_point_timeseries(sbas_dir, scene_tab, Path(gnss_file), plots_dir):
                viz_info["plots"]["gnss_points"] = str(plots_dir)
                result_msg += f"Created GNSS point time series plots in {plots_dir}\n"
            else:
                result_msg += "GNSS point time series extraction failed or skipped\n"
        else:
            print("WARNING: scene.tab not found, skipping GNSS point extraction")

    # =================================================================
    # STEP 6c: Velocity statistics report
    # =================================================================
    vel_ll_grd = sbas_dir / "vel_ll.grd"
    if vel_ll_grd.exists():
        print("-" * 50)
        print("STEP 6c: Velocity statistics report")
        print("-" * 50)

        stats_path = sbas_dir / "vel_ll_statistics.txt"
        if create_velocity_statistics_report(sbas_dir, stats_path, orbit):
            viz_info["plots"]["vel_statistics"] = str(stats_path)
            result_msg += "Created velocity statistics report\n"

    # =================================================================
    # STEP 7: Write metadata log
    # =================================================================
    print("-" * 50)
    print("STEP 7: Writing metadata log")
    print("-" * 50)

    log_path = write_meta_log(project_root, orbit, projection_info, kml_info, viz_info)
    print(f"Metadata saved to: {log_path}")
    result_msg += f"Metadata saved to {log_path}\n"

    print("=" * 70)
    print("STAGE 10 COMPLETED")
    print("=" * 70)

    return log_path, result_msg


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Command-line entry point for Stage 10 Post-SBAS."""
    parser = argparse.ArgumentParser(
        description="GMTSAR Stage 10: Post-SBAS Processing - DRAFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run post-SBAS with default settings
  python post_sbas_10_DRAFT.py /path/to/project asc

  # Run with custom velocity range
  python post_sbas_10_DRAFT.py /path/to/project asc --vel-min -50 --vel-max 50

  # Skip displacement grid projection (faster)
  python post_sbas_10_DRAFT.py /path/to/project asc --no-project-disp

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
        default="F2",
        help="Subswath for trans.dat lookup (default: F2)"
    )
    parser.add_argument(
        "--vel-cmap",
        default="jet",
        help="GMT colormap for velocity (default: jet)"
    )
    parser.add_argument(
        "--vel-min",
        type=float,
        default=None,
        help="Minimum velocity for colorbar (default: auto)"
    )
    parser.add_argument(
        "--vel-max",
        type=float,
        default=None,
        help="Maximum velocity for colorbar (default: auto)"
    )
    parser.add_argument(
        "--no-project-disp",
        action="store_true",
        help="Skip projecting displacement grids"
    )
    parser.add_argument(
        "--max-disp-grids",
        type=int,
        default=50,
        help="Maximum number of displacement grids to project (default: 50)"
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    vel_range = None
    if args.vel_min is not None and args.vel_max is not None:
        vel_range = (args.vel_min, args.vel_max)

    print(f"Project root: {project_root}")
    print(f"Orbit: {args.orbit}")
    print(f"Subswath: {args.subswath}")
    print(f"Velocity colormap: {args.vel_cmap}")
    print(f"Project displacement grids: {not args.no_project_disp}")
    print("-" * 60)

    log_path, result_msg = run_post_sbas(
        project_root=project_root,
        orbit=args.orbit,
        subswath=args.subswath,
        vel_cmap=args.vel_cmap,
        vel_range=vel_range,
        project_disp=not args.no_project_disp,
        max_disp_grids=args.max_disp_grids
    )

    print(result_msg)
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
