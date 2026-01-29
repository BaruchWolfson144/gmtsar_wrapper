#!/usr/bin/env python3
"""
make_dem_03.py - GMTSAR Step 3: DEM Creation

This script handles Digital Elevation Model (DEM) creation for InSAR processing.
It corresponds to Step 3 in the GMTSAR SBAS workflow: "Preparing the Topography DEM Grid Files"

The DEM must be created AFTER data download (Step 2) and BEFORE orbit processing (Step 4).

Usage:
    python make_dem_03.py /path/to/project_root \
        --minlon -157 --maxlon -154.2 --minlat 18 --maxlat 20.4 \
        --mode 1

Reference:
    sentinel_time_series.pdf - Section 3 (Page 3)
"""
import argparse
import subprocess
import json
from pathlib import Path
import datetime
import shutil

def run_cmd(cmd):
    """Execute shell command and return results."""
    import os
    # Preserve the full environment including PATH
    env = os.environ.copy()
    proc = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def try_make_dem_csh(project_root: Path, bbox, mode, make_dem_path=None):
    """
    Create DEM using GMTSAR's make_dem.csh script.

    Args:
        project_root: Root directory of the project
        bbox: Dictionary with minlon, maxlon, minlat, maxlat
        mode: 1 for SRTM1 (30m), 2 for SRTM3 (90m)
        make_dem_path: Optional explicit path to make_dem.csh

    Returns:
        tuple: (dem_path, message)
    """
    if make_dem_path is None:
        make_dem_path = shutil.which("make_dem.csh")
    if not make_dem_path:
        return None, "make_dem.csh not found in PATH"

    # make_dem.csh needs to run in the topo directory
    topo_dir = project_root / "asc" / "topo"
    topo_dir.mkdir(parents=True, exist_ok=True)

    # Save current directory and change to topo
    original_dir = Path.cwd()
    try:
        import os
        os.chdir(topo_dir)

        cmd = f"{make_dem_path} {bbox['minlon']} {bbox['maxlon']} {bbox['minlat']} {bbox['maxlat']} {mode}"
        rc, out, err = run_cmd(cmd)
        if rc != 0:
            return None, f"make_dem.csh failed (rc={rc}): {err.strip()}"

        # make_dem.csh creates dem.grd in the current working directory
        dem_file = topo_dir / "dem.grd"
        if not dem_file.exists():
            return None, f"make_dem.csh completed but dem.grd not found at {dem_file}"

        return dem_file, f"Created via make_dem.csh at {str(dem_file)}"

    finally:
        # Always restore original directory
        os.chdir(original_dir)


def link_dem_to_project(project_root: Path, dem_path: Path):
    """
    Link or copy DEM file to standard project locations.

    The DEM is placed in asc/topo/dem.grd and symlinked to:
    - asc/F1/topo/dem.grd
    - asc/F2/topo/dem.grd
    - asc/F3/topo/dem.grd
    - asc/merge/dem.grd
    - des/topo/dem.grd (if descending orbit exists)

    Args:
        project_root: Root directory of the project
        dem_path: Path to the source dem.grd file
    """
    dest = project_root / "asc" / "topo" / "dem.grd"
    if dem_path.resolve() != dest.resolve():
        shutil.copy2(dem_path, dest)

    targets = [
        project_root / "asc" / "F1" / "topo" / "dem.grd",
        project_root / "asc" / "F2" / "topo" / "dem.grd",
        project_root / "asc" / "F3" / "topo" / "dem.grd",
        project_root / "asc" / "merge" / "dem.grd"
    ]
    for t in targets:
        try:
            if t.exists() or t.is_symlink():
                t.unlink()
            t.symlink_to(dest)
        except Exception:
            shutil.copy2(dest, t)

    des_topo = project_root / "des" / "topo"
    if des_topo.exists():
        ddest = des_topo / "dem.grd"
        if not ddest.exists():
            shutil.copy2(dest, ddest)


def write_meta_log(project_root: Path, bbox, mode, result_msg, dem_path):
    """Write processing metadata to JSON log file."""
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": "make_dem",
        "bbox": bbox,
        "mode": mode,
        "result": result_msg,
        "dem_path": str(dem_path) if dem_path else None
    }
    logp = project_root / "wrapper_meta" / "logs" / "dem_make.json"
    logp.parent.mkdir(parents=True, exist_ok=True)
    with open(logp, "w") as f:
        json.dump(meta, f, indent=2)
    return logp


def run_make_dem(project_root: Path, bbox: dict, mode: int, make_dem_path=None):
    """
    Complete DEM creation workflow.

    Args:
        project_root: Root directory of the project
        bbox: Bounding box dictionary with minlon, maxlon, minlat, maxlat
        mode: 1 for SRTM1 (30m), 2 for SRTM3 (90m)
        make_dem_path: Optional explicit path to make_dem.csh

    Returns:
        tuple: (log_path, result_message)
    """
    dem_path, msg = try_make_dem_csh(
        project_root,
        bbox,
        mode,
        make_dem_path=make_dem_path
    )

    result_msg = "make_dem: " + msg + "\n"

    if dem_path:
        link_dem_to_project(project_root, dem_path)
        result_msg += f"Linked dem from {dem_path}\n"

    logp = write_meta_log(project_root, bbox, mode, result_msg, dem_path)
    return logp, result_msg


def main():
    parser = argparse.ArgumentParser(
        description="GMTSAR Step 3: Create DEM for InSAR processing"
    )
    parser.add_argument("project_root", help="Path to project root directory")
    parser.add_argument("--minlon", type=float, required=True, help="Minimum longitude for DEM")
    parser.add_argument("--maxlon", type=float, required=True, help="Maximum longitude for DEM")
    parser.add_argument("--minlat", type=float, required=True, help="Minimum latitude for DEM")
    parser.add_argument("--maxlat", type=float, required=True, help="Maximum latitude for DEM")
    parser.add_argument("--mode", type=int, default=1, choices=[1,2],
                       help="1 -> SRTM1 (30m), 2 -> SRTM3 (90m)")
    parser.add_argument("--make_dem_path", help="Optional explicit path to make_dem.csh")

    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    bbox = {
        "minlon": args.minlon,
        "maxlon": args.maxlon,
        "minlat": args.minlat,
        "maxlat": args.maxlat
    }

    print(f"Starting DEM creation...")
    print(f"Project root: {project_root}")
    print(f"BBox: {bbox}")
    print(f"Mode: {args.mode} ({'SRTM1 30m' if args.mode == 1 else 'SRTM3 90m'})")
    print("-" * 60)

    logp, result_msg = run_make_dem(project_root, bbox, args.mode, args.make_dem_path)
    print(result_msg)
    if logp:
        print(f"Log file: {logp}")


if __name__ == "__main__":
    main()
