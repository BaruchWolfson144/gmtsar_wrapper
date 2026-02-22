#!/usr/bin/env python3
"""
make_orbits_04.py - GMTSAR Step 3c & 4: Reframing and Orbit Download

This script handles:
1. Creating pins.ll file for reframing (Step 3c)
2. Downloading precise/restituted orbits (Step 4)
3. Optional reframing (stitching/cropping) combined with orbit download (Step 4b)

According to the GMTSAR manual (Section 4.b, pages 11-12), the organize_files_tops.csh
script can perform BOTH orbit downloading and reframing in a single workflow when
provided with both SAFE_filelist and pins.ll.

This must run AFTER DEM creation (Step 2) but BEFORE alignment (Step 5).

Usage:
    # Simple orbit download only:
    python make_orbits_04.py /path/to/project_root --mode 1

    # Combined orbit download + reframing:
    python make_orbits_04.py /path/to/project_root --mode 1 \
        --reframe --pin1_lon -157 --pin1_lat 18 --pin2_lon -154.2 --pin2_lat 20.4

Reference:
    sentinel_time_series.pdf - Section 3c (Page 9) and Section 4 (Pages 11-12)
"""
import argparse
import subprocess
import json
import shutil
import sys
import re
from pathlib import Path
from collections import defaultdict
import datetime

# Import parallel reframe module
try:
    from parallel_reframe import run_parallel_reframe
    PARALLEL_REFRAME_AVAILABLE = True
except ImportError:
    PARALLEL_REFRAME_AVAILABLE = False

def run_cmd(cmd):
    """Execute shell command and return results."""
    import os
    # Preserve the full environment including PATH
    env = os.environ.copy()
    proc = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def create_orbit_list(project_root: Path, orbit: str = "asc"):
    """
    Create SAFE_filelist file containing paths to all .SAFE files.

    Args:
        project_root: Root directory of the project
        orbit: Orbit directory (asc/des, default: asc)

    Returns:
        Path to SAFE_filelist file or None on error
    """
    data_dir = project_root / orbit / "data"
    output_file = data_dir / "SAFE_filelist"

    safe_files = list(data_dir.glob("*.SAFE"))
    if not safe_files:
        print(f"Warning: No .SAFE files found in {data_dir}")
        return None

    file_paths = [str(f.resolve()) for f in sorted(safe_files)]

    try:
        output_file.write_text('\n'.join(file_paths) + '\n', encoding='utf-8')
        print(f"Created SAFE_filelist: {output_file.resolve()}")
        return output_file.resolve()
    except Exception as e:
        print(f"Error creating SAFE_filelist: {e}")
        return None

def create_pins_file(project_root: Path, orbit: str, pin1_lon: float, pin1_lat: float,
                     pin2_lon: float, pin2_lat: float) -> Path:
    """
    Create pins.ll file for reframing (stitching or cropping frames).

    This follows Section 3c of the GMTSAR manual for adjusting region of interest.
    Pins should be ordered in the along-track direction:
    - For ascending: pin 1 (south) to pin 2 (north)
    - For descending: pin 1 (north) to pin 2 (south)

    Args:
        project_root: Root directory of the project
        orbit: Orbit directory (asc/des)
        pin1_lon: Longitude of first pin
        pin1_lat: Latitude of first pin
        pin2_lon: Longitude of second pin
        pin2_lat: Latitude of second pin

    Returns:
        Path to created pins.ll file
    """
    reframed_dir = project_root / orbit / "reframed"
    reframed_dir.mkdir(parents=True, exist_ok=True)

    pins_file = reframed_dir / "pins.ll"

    with open(pins_file, "w") as f:
        f.write(f"{pin1_lon} {pin1_lat}\n")
        f.write(f"{pin2_lon} {pin2_lat}\n")

    print(f"Created pins file: {pins_file}")
    return pins_file


def download_orbits_simple(orbit_list: Path, mode: int = 1):
    """
    Download orbits using download_sentinel_orbits_linux.csh (Section 4.a).

    This is the simple orbit download without reframing.

    Args:
        orbit_list: Path to SAFE_filelist
        mode: 1 for precise, 2 for restituted orbits

    Returns:
        tuple: (success, commands list)
    """
    import os

    # The script downloads orbit files to the current directory
    # We need to run it from the data directory
    data_dir = orbit_list.parent
    orbit_dir = data_dir.parent / "orbit"
    orbit_dir.mkdir(parents=True, exist_ok=True)

    # Save current directory
    original_dir = Path.cwd()

    try:
        # Change to orbit directory where files should be downloaded
        os.chdir(orbit_dir)

        commands = []
        cmd = f"download_sentinel_orbits_linux.csh {orbit_list} {mode}"
        rc, out, err = run_cmd(cmd)
        commands.append({
            "cmd": cmd,
            "returncode": rc,
            "stdout": out.strip(),
            "stderr": err.strip()
        })

        return rc == 0, commands

    finally:
        # Always restore original directory
        os.chdir(original_dir)


def write_meta_log(
    project_root: Path,
    orbit_list: Path,
    mode: int,
    stitching_frames: bool,
    commands,
    success: bool
):
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": "download_orbits",
        "project_root": str(project_root),
        "inputs": {
            "orbit_list": str(orbit_list),
            "mode": mode,
            "stitching_frames": stitching_frames
        },
        "commands": commands,
        "result": {
            "success": success
        }
    }

    logp = project_root / "wrapper_meta" / "logs" / "make_orbit.json"
    logp.parent.mkdir(parents=True, exist_ok=True)

    with open(logp, "w") as f:
        json.dump(meta, f, indent=2)

    return logp


def download_orbits_with_reframe(project_root: Path, orbit: str, orbit_list: Path,
                                 pins_file: Path, parallel_config: dict = None) -> tuple:
    """
    Download orbits AND reframe data using organize_files_tops_linux.csh (Section 4.b).

    This is the recommended combined workflow that performs both orbit downloading
    and reframing (stitching/cropping) in a single process.

    According to the manual (page 11-12), organize_files_tops.csh with pins.ll:
    - Mode 1: Downloads orbits and prepares files (~5 min)
    - Mode 2: Creates stitched/cropped SAFE directories (~35 min)

    With parallel_reframe enabled, mode 2 runs in parallel using directory isolation.

    Args:
        project_root: Root directory of the project
        orbit: Orbit directory (asc/des)
        orbit_list: Path to SAFE_filelist
        pins_file: Path to pins.ll file
        parallel_config: Optional dict with parallel settings:
            - parallel_reframe: bool (default: False)
            - reframe_workers: int (default: 4)

    Returns:
        tuple: (success, commands list, result message)
    """
    import os

    commands = []
    result_msg = ""

    # Get parallel settings
    parallel_config = parallel_config or {}
    use_parallel = parallel_config.get("parallel_reframe", False)
    reframe_workers = parallel_config.get("reframe_workers", 4)

    # Check if parallel reframe is available
    if use_parallel and not PARALLEL_REFRAME_AVAILABLE:
        print("Warning: parallel_reframe module not available, falling back to sequential")
        use_parallel = False

    # organize_files_tops_linux.csh creates output in current working directory
    # We need to run it from the data directory so outputs go to the right place
    data_dir = project_root / orbit / "data"
    original_dir = Path.cwd()

    try:
        os.chdir(data_dir)

        # Step 1: Run organize_files_tops_linux.csh with mode 1
        # This downloads orbits and prepares files
        cmd1 = f"organize_files_tops_linux.csh {orbit_list} {pins_file} 1"
        rc1, out1, err1 = run_cmd(cmd1)
        commands.append({
            "cmd": cmd1,
            "returncode": rc1,
            "stdout": out1.strip(),
            "stderr": err1.strip()
        })

        if rc1 != 0:
            result_msg = f"organize_files_tops mode 1 failed: {err1}"
            return False, commands, result_msg

        # Step 2: Reframing (mode 2) - either parallel or sequential
        if use_parallel:
            # =================================================================
            # PARALLEL REFRAMING
            # =================================================================
            print(f"\nUsing parallel reframing with {reframe_workers} workers...")

            # Restore directory before calling parallel_reframe
            os.chdir(original_dir)

            success, results = run_parallel_reframe(
                project_root,
                orbit,
                num_workers=reframe_workers,
                pins_file=pins_file
            )

            # Record parallel reframe in commands
            successful_dates = sum(1 for r in results if r['success'])
            failed_dates = sum(1 for r in results if not r['success'])
            total_duration = sum(r.get('duration', 0) for r in results)

            commands.append({
                "cmd": f"parallel_reframe(workers={reframe_workers})",
                "returncode": 0 if success else 1,
                "dates_processed": len(results),
                "dates_successful": successful_dates,
                "dates_failed": failed_dates,
                "total_duration": total_duration
            })

            if not success:
                failed_list = [r['date'] for r in results if not r['success']][:5]
                result_msg = f"Parallel reframing failed: {failed_dates} dates failed. First: {failed_list}"
                return False, commands, result_msg

            # Check for reframed directories
            reframed_dirs = sorted(data_dir.glob("F*_F*"))
            if reframed_dirs:
                final_dir = reframed_dirs[0]
                safe_count = len(list(final_dir.glob("*.SAFE")))
                result_msg = f"Parallel reframing successful: {successful_dates} dates processed in {total_duration:.1f}s. "
                result_msg += f"Total: {safe_count} reframed SAFE files in {final_dir.name}"
            else:
                result_msg = f"Parallel reframing completed but no F*_F* directory found"

            return success, commands, result_msg

        else:
            # =================================================================
            # SEQUENTIAL REFRAMING (original behavior)
            # =================================================================
            cmd2 = f"organize_files_tops_linux.csh {orbit_list} {pins_file} 2"
            rc2, out2, err2 = run_cmd(cmd2)
            commands.append({
                "cmd": cmd2,
                "returncode": rc2,
                "stdout": out2.strip(),
                "stderr": err2.strip()
            })

            if rc2 != 0:
                result_msg = f"organize_files_tops mode 2 failed: {err2}"
                return False, commands, result_msg

        # Check for Fxxxx_Fxxxx directories created (reframed data)
        reframed_dirs = sorted(data_dir.glob("F*_F*"))

        if not reframed_dirs:
            result_msg = "Warning: No Fxxxx_Fxxxx directories found after reframing"
            return True, commands, result_msg

        # If multiple F*_F* directories exist, merge them into one
        # (per GMTSAR manual: timing differences can create multiple directories)
        if len(reframed_dirs) > 1:
            target_dir = reframed_dirs[0]
            merged_count = 0
            for src_dir in reframed_dirs[1:]:
                # Move all SAFE directories from src to target
                for safe_dir in src_dir.glob("*.SAFE"):
                    dst = target_dir / safe_dir.name
                    if not dst.exists():
                        shutil.move(str(safe_dir), str(target_dir))
                        merged_count += 1
                # Remove empty source directory
                if not any(src_dir.iterdir()):
                    src_dir.rmdir()

            result_msg = f"Reframing successful. Merged {len(reframed_dirs)} directories into {target_dir.name} "
            result_msg += f"({merged_count} SAFE files moved)"
        else:
            result_msg = f"Reframing successful. Created directory: {reframed_dirs[0].name}"

        # Count total SAFE files in the final directory
        final_dir = sorted(data_dir.glob("F*_F*"))[0]
        safe_count = len(list(final_dir.glob("*.SAFE")))
        result_msg += f" - Total: {safe_count} reframed SAFE files"

        return True, commands, result_msg

    finally:
        # Always restore original directory
        os.chdir(original_dir)


def validate_pins_coverage(project_root: Path, orbit: str,
                           pin1_lat: float, pin2_lat: float,
                           margin: float = 0.1) -> dict:
    """
    Validate that pins are within the coverage of ALL SAFE frames.

    This prevents the issue where pins extend beyond some frames' footprints,
    causing those dates to fail during reframing.

    Args:
        project_root: Root directory of the project
        orbit: Orbit directory (asc/des)
        pin1_lat, pin2_lat: Pin latitudes
        margin: Safety margin in degrees (default: 0.1)

    Returns:
        dict with 'valid' boolean and 'message' string
    """
    data_dir = project_root / orbit / "data"
    south_pin = min(pin1_lat, pin2_lat)
    north_pin = max(pin1_lat, pin2_lat)

    # Scan all SAFE directories for footprints
    safe_dirs = sorted(data_dir.glob("S1*.SAFE"))
    if not safe_dirs:
        return {'valid': False, 'message': f"No SAFE directories found in {data_dir}"}

    footprints_by_date = defaultdict(list)

    for safe_dir in safe_dirs:
        manifest = safe_dir / "manifest.safe"
        if not manifest.exists():
            continue

        try:
            content = manifest.read_text()
            match = re.search(r'<gml:coordinates>([^<]+)</gml:coordinates>', content)
            if not match:
                continue

            coords = []
            for pair in match.group(1).strip().split():
                lat, lon = pair.split(',')
                coords.append(float(lat))

            min_lat = min(coords)
            max_lat = max(coords)

            # Extract date from SAFE name
            date_match = re.search(r'_(\d{8})T', safe_dir.name)
            date = date_match.group(1) if date_match else safe_dir.name
            footprints_by_date[date].append({'min_lat': min_lat, 'max_lat': max_lat})
        except Exception:
            continue

    if not footprints_by_date:
        return {'valid': False, 'message': "Could not parse any SAFE footprints"}

    # Calculate safe coverage (intersection of all dates)
    date_coverages = {}
    for date, frames in footprints_by_date.items():
        combined_min = min(f['min_lat'] for f in frames)
        combined_max = max(f['max_lat'] for f in frames)
        date_coverages[date] = {'min_lat': combined_min, 'max_lat': combined_max}

    safe_min_lat = max(dc['min_lat'] for dc in date_coverages.values())
    safe_max_lat = min(dc['max_lat'] for dc in date_coverages.values())

    issues = []

    # Check southern pin
    if south_pin < safe_min_lat:
        diff = safe_min_lat - south_pin
        # Find specific dates that would fail (their coverage doesn't reach the southern pin)
        failed_dates_south = sorted([d for d, dc in date_coverages.items()
                                      if dc['min_lat'] > south_pin])
        issues.append(f"Southern pin ({south_pin:.4f}°) is {diff:.4f}° below safe minimum ({safe_min_lat:.4f}°). "
                     f"{len(failed_dates_south)} dates would fail!")
        if failed_dates_south:
            print(f"\n  SCENES FAILING SOUTHERN PIN CHECK ({len(failed_dates_south)} dates):")
            for d in failed_dates_south:
                dc = date_coverages[d]
                print(f"    - {d}: coverage {dc['min_lat']:.4f}° to {dc['max_lat']:.4f}°")

    # Check northern pin
    if north_pin > safe_max_lat:
        diff = north_pin - safe_max_lat
        # Find specific dates that would fail (their coverage doesn't reach the northern pin)
        failed_dates_north = sorted([d for d, dc in date_coverages.items()
                                      if dc['max_lat'] < north_pin])
        issues.append(f"Northern pin ({north_pin:.4f}°) is {diff:.4f}° above safe maximum ({safe_max_lat:.4f}°). "
                     f"{len(failed_dates_north)} dates would fail!")
        if failed_dates_north:
            print(f"\n  SCENES FAILING NORTHERN PIN CHECK ({len(failed_dates_north)} dates):")
            for d in failed_dates_north:
                dc = date_coverages[d]
                print(f"    - {d}: coverage {dc['min_lat']:.4f}° to {dc['max_lat']:.4f}°")

    if issues:
        return {
            'valid': False,
            'message': "PINS VALIDATION FAILED:\n" + "\n".join(f"  - {i}" for i in issues) +
                       f"\n\nSuggested safe range: {safe_min_lat + margin:.2f}° to {safe_max_lat - margin:.2f}°",
            'safe_min': safe_min_lat,
            'safe_max': safe_max_lat,
            'num_dates': len(date_coverages)
        }

    return {
        'valid': True,
        'message': f"Pins validated: within coverage of all {len(date_coverages)} dates "
                   f"(safe range: {safe_min_lat:.4f}° to {safe_max_lat:.4f}°)",
        'safe_min': safe_min_lat,
        'safe_max': safe_max_lat,
        'num_dates': len(date_coverages)
    }


def run_download_orbits(
    project_root: Path,
    orbit: str = "asc",
    mode: int = 1,
    reframe: bool = False,
    pin1_lon: float = None,
    pin1_lat: float = None,
    pin2_lon: float = None,
    pin2_lat: float = None,
    parallel_config: dict = None
):
    """
    Complete workflow for orbit download with optional reframing.

    Args:
        project_root: Root directory of the project
        orbit: Orbit directory (asc/des, default: asc)
        mode: 1 for precise, 2 for restituted orbits
        reframe: If True, perform reframing with orbit download
        pin1_lon, pin1_lat: First pin coordinates (required if reframe=True)
        pin2_lon, pin2_lat: Second pin coordinates (required if reframe=True)
        parallel_config: Optional dict with parallel settings:
            - parallel_reframe: bool - enable parallel reframing
            - reframe_workers: int - number of parallel workers

    Returns:
        tuple: (success, result_message, log_path)
    """
    # Create SAFE_filelist
    orbit_list = create_orbit_list(project_root, orbit)
    if not orbit_list:
        return False, "Failed to create SAFE_filelist", None

    commands = []
    result_msg = ""

    if reframe:
        # Validate pin coordinates
        if not all([pin1_lon, pin1_lat, pin2_lon, pin2_lat]):
            return False, "All pin coordinates required for reframing", None

        # Pre-check: Validate pins against SAFE footprints
        print("Validating pins against SAFE footprints...")
        validation = validate_pins_coverage(project_root, orbit, pin1_lat, pin2_lat)
        print(f"  {validation['message']}")
        if not validation['valid']:
            return False, validation['message'], None

        # Create pins.ll file
        pins_file = create_pins_file(project_root, orbit, pin1_lon, pin1_lat,
                                     pin2_lon, pin2_lat)

        # Combined orbit download + reframing (Section 4.b)
        success, commands, result_msg = download_orbits_with_reframe(
            project_root, orbit, orbit_list, pins_file, parallel_config
        )
    else:
        # Simple orbit download only (Section 4.a)
        success, commands = download_orbits_simple(orbit_list, mode)
        result_msg = "Orbit download " + ("completed successfully" if success else "failed")

    # Write metadata log
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": "download_orbits" + ("_with_reframe" if reframe else ""),
        "orbit": orbit,
        "project_root": str(project_root),
        "inputs": {
            "orbit_list": str(orbit_list),
            "mode": mode,
            "reframe": reframe
        },
        "commands": commands,
        "result": {
            "success": success,
            "message": result_msg
        }
    }

    if reframe:
        meta["inputs"]["pins"] = {
            "pin1": {"lon": pin1_lon, "lat": pin1_lat},
            "pin2": {"lon": pin2_lon, "lat": pin2_lat}
        }
        if parallel_config:
            meta["inputs"]["parallel_config"] = parallel_config

    logp = project_root / "wrapper_meta" / "logs" / f"orbits_{orbit}.json"
    logp.parent.mkdir(parents=True, exist_ok=True)

    with open(logp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return success, result_msg, logp

def main():
    parser = argparse.ArgumentParser(
        description="GMTSAR Step 3c & 4: Download orbits with optional reframing"
    )
    parser.add_argument("project_root", type=str, help="Path to project root directory")
    parser.add_argument("--orbit", type=str, default="asc",
                       help="Orbit directory (asc/des, default: asc)")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2],
                       help="1 for precise orbits, 2 for restituted orbits")

    # Reframe arguments
    parser.add_argument("--reframe", action="store_true",
                       help="Perform reframing (stitching or cropping) with orbit download")
    parser.add_argument("--pin1_lon", type=float,
                       help="Longitude of first pin for reframing")
    parser.add_argument("--pin1_lat", type=float,
                       help="Latitude of first pin for reframing")
    parser.add_argument("--pin2_lon", type=float,
                       help="Longitude of second pin for reframing")
    parser.add_argument("--pin2_lat", type=float,
                       help="Latitude of second pin for reframing")

    # Parallel reframing arguments
    parser.add_argument("--parallel-reframe", action="store_true",
                       help="Use parallel reframing (requires reframe flag)")
    parser.add_argument("--reframe-workers", type=int, default=4,
                       help="Number of parallel workers for reframing (default: 4)")

    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    # Validate reframe arguments
    if args.reframe:
        if not all([args.pin1_lon, args.pin1_lat, args.pin2_lon, args.pin2_lat]):
            print("Error: All pin coordinates must be provided for reframing")
            print("Required: --pin1_lon, --pin1_lat, --pin2_lon, --pin2_lat")
            return

    print(f"{'=' * 60}")
    print(f"GMTSAR Step 3c & 4: Orbit Download" + (" + Reframing" if args.reframe else ""))
    print(f"{'=' * 60}")
    print(f"Project root: {project_root}")
    print(f"Orbit: {args.orbit}")
    print(f"Mode: {args.mode} ({'precise' if args.mode == 1 else 'restituted'})")

    # Build parallel config
    parallel_config = None
    if args.reframe:
        print(f"Reframing: YES")
        print(f"Pin 1: ({args.pin1_lon}, {args.pin1_lat})")
        print(f"Pin 2: ({args.pin2_lon}, {args.pin2_lat})")
        if args.parallel_reframe:
            parallel_config = {
                "parallel_reframe": True,
                "reframe_workers": args.reframe_workers
            }
            print(f"Parallel reframe: YES ({args.reframe_workers} workers)")
        else:
            print(f"Parallel reframe: NO (sequential)")
    else:
        print(f"Reframing: NO")
    print("-" * 60)

    success, result_msg, logp = run_download_orbits(
        project_root,
        orbit=args.orbit,
        mode=args.mode,
        reframe=args.reframe,
        pin1_lon=args.pin1_lon,
        pin1_lat=args.pin1_lat,
        pin2_lon=args.pin2_lon,
        pin2_lat=args.pin2_lat,
        parallel_config=parallel_config
    )

    print("\n" + result_msg)
    if logp:
        print(f"Log file: {logp}")

    if not success:
        print("\nERROR: Processing failed. Check log file for details.")
        exit(1)

if __name__ == "__main__":
    main()