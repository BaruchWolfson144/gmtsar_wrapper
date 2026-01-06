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
from pathlib import Path
import datetime

def run_cmd(cmd):
    """Execute shell command and return results."""
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
                                 pins_file: Path) -> tuple:
    """
    Download orbits AND reframe data using organize_files_tops_linux.csh (Section 4.b).

    This is the recommended combined workflow that performs both orbit downloading
    and reframing (stitching/cropping) in a single process.

    According to the manual (page 11-12), organize_files_tops.csh with pins.ll:
    - Mode 1: Downloads orbits and prepares files (~5 min)
    - Mode 2: Creates stitched/cropped SAFE directories (~35 min)

    Args:
        project_root: Root directory of the project
        orbit: Orbit directory (asc/des)
        orbit_list: Path to SAFE_filelist
        pins_file: Path to pins.ll file

    Returns:
        tuple: (success, commands list, result message)
    """
    commands = []
    result_msg = ""

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

    # Step 2: Run organize_files_tops_linux.csh with mode 2
    # This creates stitched/cropped SAFE directories
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
    data_dir = project_root / orbit / "data"
    reframed_dirs = list(data_dir.glob("F*_F*"))

    if reframed_dirs:
        result_msg = f"Reframing successful. Created {len(reframed_dirs)} reframed directories: "
        result_msg += ", ".join([d.name for d in reframed_dirs])
    else:
        result_msg = "Warning: No Fxxxx_Fxxxx directories found after reframing"

    return True, commands, result_msg


def run_download_orbits(
    project_root: Path,
    orbit: str = "asc",
    mode: int = 1,
    reframe: bool = False,
    pin1_lon: float = None,
    pin1_lat: float = None,
    pin2_lon: float = None,
    pin2_lat: float = None
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

        # Create pins.ll file
        pins_file = create_pins_file(project_root, orbit, pin1_lon, pin1_lat,
                                     pin2_lon, pin2_lat)

        # Combined orbit download + reframing (Section 4.b)
        success, commands, result_msg = download_orbits_with_reframe(
            project_root, orbit, orbit_list, pins_file
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

    if args.reframe:
        print(f"Reframing: YES")
        print(f"Pin 1: ({args.pin1_lon}, {args.pin1_lat})")
        print(f"Pin 2: ({args.pin2_lon}, {args.pin2_lat})")
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
        pin2_lat=args.pin2_lat
    )

    print("\n" + result_msg)
    if logp:
        print(f"Log file: {logp}")

    if not success:
        print("\nERROR: Processing failed. Check log file for details.")
        exit(1)

if __name__ == "__main__":
    main()