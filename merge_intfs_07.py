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
import logging

logger = logging.getLogger(__name__)


def run_cmd(cmd, cwd=None):
    import os
    # Preserve the full environment including PATH
    env = os.environ.copy()
    proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def prepare_merge_list(project_root: Path, orbit: str, master: str, mode=0):
    """
    Create merge_list and promote master to first line.

    Args:
        project_root: Project root directory
        orbit: Orbit directory (asc/des)
        master: Master stem name (e.g., "S1_20200104_ALL_F1")
        mode: Merge mode (1/2/3)

    The master format is: S1_YYYYMMDD_ALL_F#
    We extract the date (YYYYMMDD) to match against merge_list entries.
    """
    # create merge_list
    cwd = project_root / orbit / "merge"
    if not cwd.exists():
        raise RuntimeError(f"merge directory not found: {cwd}")

    # --- STEP 1: Create intflist ---
    logger.info("-" * 50)
    logger.info("STEP 1: Creating intflist from intf_all directories")
    logger.info("-" * 50)

    # Use intersection of subswaths being merged to avoid missing directories
    # mode 0: F1+F2+F3, mode 1: F1+F2, mode 2: F2+F3
    mode_labels = {0: "F1+F2+F3", 1: "F1+F2", 2: "F2+F3"}
    if mode == 0:
        subswaths = ["F1", "F2", "F3"]
    elif mode == 1:
        subswaths = ["F1", "F2"]
    else:
        subswaths = ["F2", "F3"]

    logger.info(f"  Merge mode: {mode} ({mode_labels.get(mode, '?')})")

    dir_sets = []
    for sw in subswaths:
        intf_all_dir = project_root / orbit / sw / "intf_all"
        if not intf_all_dir.exists():
            raise RuntimeError(f"intf_all directory not found: {intf_all_dir}")
        sw_dirs = set(d.name for d in intf_all_dir.iterdir() if d.is_dir())
        dir_sets.append(sw_dirs)
        logger.info(f"  [{sw}] Found {len(sw_dirs)} interferograms in intf_all")

    intf_dirs = sorted(set.intersection(*dir_sets))
    if not intf_dirs:
        raise RuntimeError(
            f"No common interferogram directories found across subswaths {subswaths}. "
            f"Counts: {', '.join(f'{sw}={len(s)}' for sw, s in zip(subswaths, dir_sets))}"
        )

    logger.info(f"  Common interferograms across {'+'.join(subswaths)}: {len(intf_dirs)}")
    intflist_path = cwd / "intflist"
    intflist_path.write_text("\n".join(intf_dirs) + "\n")

    # --- STEP 2: Running create_merge_input.csh ---
    logger.info("-" * 50)
    logger.info("STEP 2: Running create_merge_input.csh")
    logger.info("-" * 50)

    cmd = f"create_merge_input.csh intflist .. {mode} > merge_list"
    logger.info(f"  Command: {cmd}")
    pc, out, err = run_cmd(cmd, cwd)

    if pc != 0:
        logger.warning(f"  create_merge_input.csh returned code {pc}")
        if err.strip():
            logger.warning(f"  stderr: {err.strip()[:200]}")

    # --- STEP 3: Promoting master to first line ---
    logger.info("-" * 50)
    logger.info("STEP 3: Promoting master image to first line in merge_list")
    logger.info("-" * 50)

    merge_list = cwd / "merge_list"
    lines = []

    # Extract date from master stem: S1_YYYYMMDD_ALL_F# -> YYYYMMDD (8 digits)
    master_base = master[3:11]  # "S1_20200104_ALL_F1" -> "20200104"
    master_pattern = re.compile(rf"S[1-3]_{master_base}")
    master_line = None

    logger.info(f"  Master: {master} (date: {master_base})")

    merge_list_content = merge_list.read_text().strip()
    if not merge_list_content:
        raise RuntimeError(
            f"merge_list is empty. This usually means no interferograms were created. "
            f"Check that intflist exists and contains interferogram pairs."
        )

    for line in merge_list_content.splitlines():
        if not line.strip():
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        second = parts[1]
        if master_pattern.match(second):
            master_line = line
        else:
            lines.append(line)

    if master_line is None:
        raise RuntimeError(
            f"Master line not found in merge_list. "
            f"Master: {master}, extracted date: {master_base}, pattern: {master_pattern.pattern}"
        )

    merge_list.write_text(master_line + "\n" + "\n".join(lines))
    logger.info(f"  merge_list created with {1 + len(lines)} entries (master promoted to first line)")

    meta = {
        "orbit": orbit,
        "command": cmd,
        "mode": mode,
        "master": master,
        "master_base": master_base,
        "num_total_lines": 1 + len(lines),
        "master_promoted": True,
        "return_code": pc,
        "stderr": err.strip(),
    }
    return meta

def prepare_merge_run(project_root: Path, orbit: str):
    # --- STEP 4: Preparing merge run files ---
    logger.info("-" * 50)
    logger.info("STEP 4: Preparing merge configuration files")
    logger.info("-" * 50)

    merge_dir = project_root / orbit / "merge"

    # copy config
    src_config = project_root / orbit / "F1" / "batch_tops.config"
    dst_config = merge_dir / "batch_tops.config"
    shutil.copy2(src_config, dst_config)
    logger.info(f"  Copied batch_tops.config from F1 to merge/")

    # make sure that threshold_snaphu = 0
    lines = []
    for line in dst_config.read_text().splitlines():
        if line.strip().startswith("threshold_snaphu"):
            line = "threshold_snaphu = 0"
        lines.append(line)
    dst_config.write_text("\n".join(lines) + "\n")
    logger.info("  Set threshold_snaphu = 0 (unwrapping will be done after merge)")

    # link dem.grd
    dem_src = project_root / orbit / "topo" / "dem.grd"
    dem_link = merge_dir / "dem.grd"

    if not dem_link.exists():
        dem_link.symlink_to(dem_src)
        logger.info(f"  Linked dem.grd -> {dem_src}")
    else:
        logger.info(f"  dem.grd link already exists")

    meta = {
        "orbit": orbit,
        "merge_dir": str(merge_dir),
        "config_source": str(src_config),
        "config_target": str(dst_config),
        "forced_parameters": {
            "threshold_snaphu": 0,
        },
        "link_dem": "ok"
    }
    return meta

def merge(project_root: Path, orbit: str):
    # --- STEP 5: Running merge_batch.csh ---
    logger.info("-" * 50)
    logger.info("STEP 5: Running merge_batch.csh (with resume)")
    logger.info("-" * 50)

    cwd = project_root / orbit / "merge"

    # Resume: filter merge_list to skip already-merged interferograms
    # The first line (master) must always be kept as reference
    merge_list_path = cwd / "merge_list"
    all_lines = [l for l in merge_list_path.read_text().strip().splitlines() if l.strip()]

    if not all_lines:
        raise RuntimeError("merge_list is empty")

    master_line = all_lines[0]
    remaining_lines = all_lines[1:]

    # Check which interferograms already have phasefilt.grd in merge/
    already_merged = 0
    new_lines = []
    for line in remaining_lines:
        # Extract dir name from merge_list line:
        # ../F1/intf_all/2020001_2020013/:S1...:S1...,../F2/...
        # The dir name is the intf pair like 2020001_2020013
        first_path = line.split(",")[0].split(":")[0]  # e.g. ../F1/intf_all/2020001_2020013/
        dir_name = Path(first_path).name  # e.g. 2020001_2020013
        if not dir_name:
            dir_name = Path(first_path).parent.name

        merged_dir = cwd / dir_name
        if (merged_dir / "phasefilt.grd").exists():
            already_merged += 1
        else:
            new_lines.append(line)

    if already_merged > 0:
        logger.info(f"  RESUME: {already_merged}/{len(remaining_lines)} interferograms already merged, will be skipped")

    if not new_lines:
        logger.info("  All interferograms already merged, nothing to do")
        merged_dirs = [d for d in cwd.iterdir() if d.is_dir() and "_" in d.name and d.name[0].isdigit()]
        meta = {
            "orbit": orbit,
            "working_dir": str(cwd),
            "command": "skipped (all merged)",
            "return_code": 0,
            "already_merged": already_merged,
            "newly_merged": 0,
            "total_merged": len(merged_dirs),
        }
        return meta

    # Write filtered merge_list (master line + new lines only)
    merge_list_path.write_text(master_line + "\n" + "\n".join(new_lines) + "\n")
    logger.info(f"  merge_list: {len(new_lines)} new interferograms to merge (+ master reference line)")

    cmd = "merge_batch.csh merge_list batch_tops.config"
    logger.info(f"  Command: {cmd}")
    logger.info(f"  Working directory: {cwd}")
    logger.info("  This may take a while...")
    pc, out, err = run_cmd(cmd, cwd)

    if pc == 0:
        merged_dirs = [d for d in cwd.iterdir() if d.is_dir() and "_" in d.name and d.name[0].isdigit()]
        logger.info(f"  merge_batch.csh completed successfully ({len(merged_dirs)} total merged interferograms)")
    else:
        logger.error(f"  merge_batch.csh failed with return code {pc}")
        if err.strip():
            logger.error(f"  stderr: {err.strip()[:500]}")

    meta = {
        "orbit": orbit,
        "working_dir": str(cwd),
        "command": cmd,
        "return_code": pc,
        "stdout": out.strip() if out else None,
        "stderr": err.strip() if err else None,
        "already_merged": already_merged,
        "newly_merged": len(new_lines),
    }
    return meta

def write_meta_log(project_root: Path, orbit: str, merge_list_info: dict, prepare_merge_info: dict, merge_info: dict):
    meta = {
        "step": 7,
        "orbit": orbit,
        "timestamp": datetime.datetime.now().isoformat(),
        "merge list_info": merge_list_info,
        "prepare merge_info": prepare_merge_info,
        "merge info": merge_info
    }   

    logp = (
        project_root
        / "wrapper_meta"
        / "logs"
        / f"step7_{orbit}.json"
    )
    logp.parent.mkdir(parents=True, exist_ok=True)

    with open(logp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return logp

def run_merge(project_root, orbit, master, mode):
    merge_list_info = prepare_merge_list(project_root, orbit, master, mode)
    prepare_merge_info = prepare_merge_run(project_root, orbit)
    merge_info = merge(project_root, orbit)
    return(write_meta_log(project_root, orbit, merge_list_info, prepare_merge_info, merge_info))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=str, help="path to project root")
    parser.add_argument("orbit", type=str, help="asc or dec")
    parser.add_argument("master", type=str, help="full name of master file")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=0, help="merge mode: 0=F1+F2+F3, 1=F1+F2, 2=F2+F3")
    
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    orbit = args.orbit
    master = args.master
    mode = args.mode

    run_merge(project_root, orbit, master, mode)

if __name__ == "__main__":
    main()

