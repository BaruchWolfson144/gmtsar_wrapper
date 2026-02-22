import argparse
import subprocess
import json
from pathlib import Path
import datetime
import shutil
import urllib.parse
import urllib.request
import sys
import glob
import re



def run_cmd(cmd, cwd=None):
    import os
    # Preserve the full environment including PATH
    env = os.environ.copy()
    proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def prepare_raw_links(project_root: Path, orbit: str, subswath: str):
    """
    Create symbolic links required for GMTSAR Step 5 (raw directory).

    This function links:
    - Sentinel-1 IW{n} VV XML files
    - Sentinel-1 IW{n} VV TIFF files
    - EOF orbit files
    - dem.grd

    Fallback to copy if symlink fails.
    """

    iw_map = {"F1": 1, "F2": 2, "F3": 3}
    iw = iw_map[subswath]

    raw_dir = project_root / orbit / subswath / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def link_many(pattern: str, fallback_pattern: str = None):
        files = glob.glob(pattern)

        # If no files found and fallback pattern provided, try fallback
        if not files and fallback_pattern:
            files = glob.glob(fallback_pattern)

        if not files:
            raise RuntimeError(f"No files matched: {pattern}" +
                             (f" or {fallback_pattern}" if fallback_pattern else ""))

        for f in files:
            src = Path(f)
            dst = raw_dir / src.name
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)
            except Exception:
                shutil.copy2(src, dst)

    # XML - Priority order:
    # 1. Reframed directories: F0440_F0473 (created by organize_files_tops_linux.csh)
    # 2. Single-digit frame dirs: F001, F002, F003
    # 3. Direct SAFE files in data/
    link_many(
        str(project_root / orbit / "data" / "F*_F*" / "*.SAFE" / "*" / f"*iw{iw}*vv*xml"),
        str(project_root / orbit / "data" / "*.SAFE" / "*" / f"*iw{iw}*vv*xml")
    )

    # TIFF - same priority as XML
    link_many(
        str(project_root / orbit / "data" / "F*_F*" / "*.SAFE" / "*" / f"*iw{iw}*vv*tiff"),
        str(project_root / orbit / "data" / "*.SAFE" / "*" / f"*iw{iw}*vv*tiff")
    )

    # EOF - try data dir first, then orbit dir
    link_many(
        str(project_root / orbit / "data" / "*EOF"),
        str(project_root / orbit / "orbit" / "*EOF")
    )

    # DEM
    dem_src = project_root / orbit / "topo" / "dem.grd"
    dem_dst = raw_dir / "dem.grd"

    try:
        if dem_dst.exists() or dem_dst.is_symlink():
            dem_dst.unlink()
        dem_dst.symlink_to(dem_src)
    except Exception:
        shutil.copy2(dem_src, dem_dst)


def generate_data_in(raw_dir: Path):
    cmd = 'printf "y\ny\n" | prep_data_linux.csh'
    return(run_cmd(cmd,  cwd=raw_dir))

def run_preproc_mode(raw_dir: Path, mode: int):
    """
    Run preproc_batch_tops.csh in the specified mode.

    Args:
        raw_dir: Directory containing data.in, dem.grd, and raw data files
        mode: Preprocessing mode (1 or 2)
              Mode 1: Creates baseline_table.dat and performs initial alignment
              Mode 2: Performs final preprocessing with master as reference

    Returns:
        tuple: (returncode, stdout, stderr)
    """
    cmd = f"preproc_batch_tops.csh data.in dem.grd {mode}"
    rc, stdout, stderr = run_cmd(cmd, cwd=raw_dir)

    # Write output to log file
    log_file = raw_dir / f"pbt_mode{mode}.log"
    with open(log_file, "w") as f:
        f.write(f"Command: {cmd}\n")
        f.write(f"Return code: {rc}\n\n")
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(stderr)

    return rc, stdout, stderr

def select_master(raw_dir: Path):
    baseline_table = raw_dir / "baseline_table.dat"
    with open(baseline_table) as f:
        rows = [(p[0], float(p[2]), float(p[4])) for p in (line.split() for line in f)]
    time_mean = sum(t[1] for t in rows) / len(rows)
    perp_mean = sum(b[2] for b in rows) / len(rows)
    def dist2(row):
        _, t, b = row
        return (t - time_mean)**2 + (b - perp_mean)**2
    master = min(rows, key=dist2)[0]
    return {
        "master": master,
        "method of choosing": "time+baseline centroid",
        "baseline table lines count": len(rows)
    }

def master_to_first_line(raw_dir: Path, master):
    """
    Reorder data.in so the master scene's raw filename entry is first.

    The master parameter is a stem name like "S1_20200104_ALL_F1" from baseline_table.dat.
    We need to find the corresponding raw filename line in data.in which contains
    the same date (e.g., "s1a-iw1-slc-vv-20200104t043025-...").

    Args:
        raw_dir: Directory containing data.in
        master: Stem name of master scene (e.g., "S1_20200104_ALL_F1")

    Returns:
        0 on success

    Raises:
        ValueError: If master's date cannot be found in data.in
    """
    baseline_table = raw_dir / "baseline_table.dat"
    data_file = raw_dir / "data.in"

    # Extract date from master stem (format: S1_YYYYMMDD_ALL_F#)
    # Example: S1_20200104_ALL_F1 -> 20200104
    date_match = re.search(r'_(\d{8})_', master)
    if not date_match:
        raise ValueError(f"Could not extract date from master stem: {master}")

    master_date = date_match.group(1)

    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Find the line containing the master date
    master_line = None
    for line in lines:
        if master_date in line:
            master_line = line
            break

    if master_line is None:
        raise ValueError(
            f"Master date '{master_date}' (from stem '{master}') "
            f"not found in {data_file}"
        )

    # Reorder: master line first, then all other lines
    new_lines = [master_line] + [l for l in lines if l != master_line]

    with open(data_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    return 0
        
        


def write_meta_log(project_root: Path, orbit: str, subswath: str, data_in_info: dict, master_info : dict, preproc_info: list, master_promoted: str):

    meta = {
        "step": 5,
        "orbit": orbit,
        "subswath": subswath,
        "timestamp": datetime.datetime.now().isoformat(),
        "data_in": {"status": data_in_info},
        "master_selection": master_info,
        "preproc": {
            "mode1": preproc_info[0],
            "mode2": preproc_info[1]
        },
        "master_promotion": master_promoted
    }


    logp = (
        project_root
        / "wrapper_meta"
        / "logs"
        / f"step5_{orbit}_{subswath}.json"
    )
    logp.parent.mkdir(parents=True, exist_ok=True)

    with open(logp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return logp



def get_master_date(master_stem: str) -> str:
    """
    Extract date from master stem.

    Args:
        master_stem: e.g., "S1_20200104_ALL_F1"

    Returns:
        Date string, e.g., "20200104"
    """
    date_match = re.search(r'_(\d{8})_', master_stem)
    if not date_match:
        raise ValueError(f"Could not extract date from master stem: {master_stem}")
    return date_match.group(1)


def is_mode1_complete(project_root: Path, orbit: str, subswath: str) -> bool:
    """
    Check if MODE 1 completed for this subswath.

    MODE 1 completion is indicated by baseline_table.dat existing in the
    parent directory (moved there after MODE 1 completes).
    """
    parent_baseline = project_root / orbit / subswath / "baseline_table.dat"
    return parent_baseline.exists()


def is_mode2_complete(project_root: Path, orbit: str, subswath: str) -> bool:
    """
    Check if MODE 2 completed for this subswath.

    MODE 2 completion is indicated by:
    - baseline_table.dat existing in raw/ (created by MODE 2)
    - SLC count matching the number of scenes in data.in
    """
    raw_dir = project_root / orbit / subswath / "raw"
    baseline_in_raw = raw_dir / "baseline_table.dat"
    data_in = raw_dir / "data.in"

    if not baseline_in_raw.exists():
        return False

    if not data_in.exists():
        return False

    # Count SLC files and compare to expected
    slc_count = len(list(raw_dir.glob("S1_*.SLC")))
    with open(data_in) as f:
        expected_count = sum(1 for line in f if line.strip())

    return slc_count >= expected_count


def run_mode1_only(
    project_root: Path,
    orbit: str,
    subswath: str,
) -> dict:
    """
    Run only MODE 1 preprocessing (baseline calculation).

    Returns dict with:
        - master_info: result of select_master()
        - data_in_status: "ok" or "failed"
        - mode1_status: "ok" or "failed"
    """
    raw_dir = project_root / orbit / subswath / "raw"

    prepare_raw_links(project_root, orbit, subswath)

    rc, stdout, stderr = generate_data_in(raw_dir)
    data_in_status = "ok" if rc == 0 else "failed"
    if rc != 0:
        raise RuntimeError(f"[{subswath}] Failed to generate data.in file. Error: {stderr}")

    rc, stdout, stderr = run_preproc_mode(raw_dir, mode=1)
    mode1_status = "ok" if rc == 0 else "failed"
    if rc != 0:
        raise RuntimeError(f"[{subswath}] Preprocessing mode 1 failed. Error: {stderr}")

    # Verify baseline_table.dat was created
    baseline_table = raw_dir / "baseline_table.dat"
    if not baseline_table.exists():
        raise FileNotFoundError(
            f"[{subswath}] baseline_table.dat not found after preprocessing mode 1. "
            f"Expected at: {baseline_table}"
        )

    master_info = select_master(raw_dir)

    # Move baseline_table.dat to parent directory (per PDF manual step 5b.ii)
    # This preserves it before MODE 2 overwrites it
    parent_baseline = raw_dir.parent / "baseline_table.dat"
    shutil.move(str(baseline_table), str(parent_baseline))

    return {
        "subswath": subswath,
        "master_info": master_info,
        "data_in_status": data_in_status,
        "mode1_status": mode1_status,
        "baseline_table_path": str(parent_baseline),
    }


def validate_master_across_subswaths(
    project_root: Path,
    orbit: str,
    subswath_list: list,
    master_date: str,
    mode1_results: dict,
) -> None:
    """
    Validate that master date exists in all subswaths and that all would select the same date.

    Args:
        project_root: Project root path
        orbit: "asc" or "dec"
        subswath_list: List of subswaths ["F1", "F2", "F3"]
        master_date: The master date selected from F1 (e.g., "20200104")
        mode1_results: Dict mapping subswath -> mode1 result dict

    Raises:
        ValueError: If master date doesn't exist in a subswath or subswaths would select different dates
    """
    errors = []

    for sub in subswath_list:
        raw_dir = project_root / orbit / sub / "raw"
        baseline_table = raw_dir / "baseline_table.dat"

        if not baseline_table.exists():
            errors.append(f"[{sub}] baseline_table.dat not found at {baseline_table}")
            continue

        # Check if master date exists in this subswath
        with open(baseline_table) as f:
            dates_in_subswath = [get_master_date(line.split()[0]) for line in f]

        if master_date not in dates_in_subswath:
            errors.append(
                f"[{sub}] Master date {master_date} not found in baseline_table.dat. "
                f"Available dates: {sorted(set(dates_in_subswath))}"
            )
            continue

        # Check if this subswath would have selected a different master
        sub_master_info = mode1_results[sub]["master_info"]
        sub_master_date = get_master_date(sub_master_info["master"])

        if sub_master_date != master_date:
            errors.append(
                f"[{sub}] Would have selected different master date: {sub_master_date} "
                f"(F1 selected: {master_date}). This indicates data inconsistency between subswaths."
            )

    if errors:
        raise ValueError(
            "Master validation failed across subswaths:\n" + "\n".join(errors)
        )


def promote_master_and_run_mode2(
    project_root: Path,
    orbit: str,
    subswath: str,
    master_date: str,
) -> dict:
    """
    Promote master to first line and run MODE 2 preprocessing.

    Args:
        project_root: Project root path
        orbit: "asc" or "dec"
        subswath: "F1", "F2", or "F3"
        master_date: The master date to use (e.g., "20200104")

    Returns:
        dict with mode2_status and master_promoted status
    """
    raw_dir = project_root / orbit / subswath / "raw"

    # Create master stem for this subswath
    # Format: S1_YYYYMMDD_ALL_F#
    master_stem = f"S1_{master_date}_ALL_{subswath}"

    r = master_to_first_line(raw_dir, master_stem)
    master_promoted = "ok" if r == 0 else "failed"
    if r != 0:
        raise RuntimeError(f"[{subswath}] Failed to promote master to first line in data.in")

    rc, stdout, stderr = run_preproc_mode(raw_dir, mode=2)
    mode2_status = "ok" if rc == 0 else "failed"
    if rc != 0:
        raise RuntimeError(f"[{subswath}] Preprocessing mode 2 failed. Error: {stderr}")

    return {
        "subswath": subswath,
        "master_promoted": master_promoted,
        "mode2_status": mode2_status,
    }


def run_preprocess_subswath(
    project_root: Path,
    orbit: str,
    subswath: str,
):
    """
    Original function for backward compatibility (sequential processing).
    For parallel processing with consistent master, use run_mode1_only +
    validate_master_across_subswaths + promote_master_and_run_mode2.
    """
    raw_dir = project_root / orbit / subswath / "raw"

    prepare_raw_links(project_root, orbit, subswath)

    rc, stdout, stderr = generate_data_in(raw_dir)

    data_in_info = "ok" if rc == 0 else "failed"
    if rc != 0:
        raise RuntimeError(f"Failed to generate data.in file. Error: {stderr}")

    rc, stdout, stderr = run_preproc_mode(raw_dir, mode=1)
    preproc_1 = "ok" if rc == 0 else "failed"
    if rc != 0:
        raise RuntimeError(f"Preprocessing mode 1 failed. Error: {stderr}")

    # Verify baseline_table.dat was created
    baseline_table = raw_dir / "baseline_table.dat"
    if not baseline_table.exists():
        raise FileNotFoundError(
            f"baseline_table.dat not found after preprocessing mode 1. "
            f"Expected at: {baseline_table}"
        )

    master_info = select_master(raw_dir)

    # Move baseline_table.dat to parent directory (per PDF manual step 5b.ii)
    # This preserves it before MODE 2 overwrites it
    parent_baseline = raw_dir.parent / "baseline_table.dat"
    shutil.move(str(baseline_table), str(parent_baseline))

    r = master_to_first_line(raw_dir, master_info["master"])
    master_promoted = "ok" if r == 0 else "failed"
    if r != 0:
        raise RuntimeError("Failed to promote master to first line in data.in")

    rc, stdout, stderr = run_preproc_mode(raw_dir, mode=2)
    preproc_2 = "ok" if rc == 0 else "failed"
    if rc != 0:
        raise RuntimeError(f"Preprocessing mode 2 failed. Error: {stderr}")

    return(write_meta_log(
        project_root,
        orbit,
        subswath,
        data_in_info,
        master_info,
        [preproc_1, preproc_2],
        master_promoted
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=str, help="path to project root")
    parser.add_argument("orbit", type=str, default="asc", help="asc or dec")
    parser.add_argument("subswath", type=str, default="F1", help="subswath to work on (F1/F2/F3)")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    orbit = args.orbit
    subswath = args.subswath

    run_preprocess_subswath(project_root, orbit, subswath)

if __name__ == "__main__":
    main()
