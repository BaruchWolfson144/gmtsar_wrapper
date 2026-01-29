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
import sys


def run_cmd(cmd, cwd=None):
    import os
    # Preserve the full environment including PATH
    env = os.environ.copy()
    proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def prepare_merge_list(project_root: Path, orbit: str, master: str, mode=2):
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
    cmd = f"create_merge_input.csh intflist .. {mode} > merge_list"
    pc, out, err = run_cmd(cmd, cwd)

    # promote master line to first line
    merge_list = cwd / "merge_list"
    lines = []

    # Extract date from master stem: S1_YYYYMMDD_ALL_F# -> YYYYMMDD (8 digits)
    # Fixed: was master[3:10] (7 chars), now master[3:11] (8 chars)
    master_base = master[3:11]  # "S1_20200104_ALL_F1" -> "20200104"
    master_pattern = re.compile(rf"S[1-3]_{master_base}")
    master_line = None

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
    merge_dir = project_root / orbit / "merge"

    # copy config 
    src_config = project_root / orbit / "F1" / "batch_tops.config"
    dst_config = merge_dir / "batch_tops.config"
    shutil.copy2(src_config, dst_config)

    # make sure that threshold_snaphu = 0
    lines = []
    for line in dst_config.read_text().splitlines():
        if line.strip().startswith("threshold_snaphu"):
            line = "threshold_snaphu = 0"
        lines.append(line)
    dst_config.write_text("\n".join(lines) + "\n")

    # link dem.grd
    dem_src = project_root / orbit / "topo" / "dem.grd"
    dem_link = merge_dir / "dem.grd"

    if not dem_link.exists():
        dem_link.symlink_to(dem_src)

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
    cwd = project_root / orbit / "merge"
    cmd = "merge_batch.csh merge_list batch_tops.config"
    pc, out, err = run_cmd(cmd, cwd)

    meta = {
        "orbit": orbit,
        "working_dir": str(cwd),
        "command": cmd,
        "return_code": pc,
        "stdout": out.strip(),
        "stderr": err.strip(),
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
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=2, help="choose mode (1/2/3) to detrmine wich subs to merge")
    
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    orbit = args.orbit
    master = args.master
    mode = args.mode

    run_merge(project_root, orbit, master, mode)

if __name__ == "__main__":
    main()

