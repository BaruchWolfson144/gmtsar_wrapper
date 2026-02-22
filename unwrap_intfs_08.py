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


def run_cmd(cmd, cwd=None, extra_path=None):
    import os
    # Preserve the full environment including PATH
    env = os.environ.copy()
    if extra_path:
        env["PATH"] = str(extra_path) + ":" + env.get("PATH", "")
    proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def make_landmask(project_root: Path, orbit: Path):
    logger.info("  Creating landmask from phasefilt.grd bounds...")
    merge_root = project_root / orbit / "merge"

    merge_intf_dir = None
    for d in sorted(merge_root.iterdir()):
        if d.is_dir() and "_" in d.name and d.name[0:2] == "20" and (d / "phasefilt.grd").exists():
            merge_intf_dir = d
            break

    if merge_intf_dir is None:
        raise RuntimeError("No merged interferogram directory found")

    cmd = "gmt grdinfo -C phasefilt.grd"
    rc, out, err = run_cmd(cmd, cwd=merge_intf_dir)

    if rc != 0:
        raise RuntimeError(f"grdinfo failed:\n{err}")

    fields = out.strip().split()
    xmin, xmax, ymin, ymax = fields[1:5]

    bounds = f"{xmin}/{xmax}/{ymin}/{ymax}"
    logger.info(f"  Running landmask.csh with bounds: {bounds}")

    rc, out, err = run_cmd(
        f"landmask.csh {bounds}",
        cwd=merge_root
    )

    if rc != 0:
        raise RuntimeError(f"landmask failed:\n{err}")

    logger.info("  landmask_ra.grd created successfully")

    meta =  {
        "grid_bounds": {
            "xmin":xmin ,
            "xmax":xmax ,
            "ymin":ymin ,
            "ymax":ymax
            },
        "reference": {
            "merge_interferogram": merge_intf_dir.name,
            "grid_file": "phasefilt.grd",
        }
    }
    return meta

def make_mask_def(project_root: Path, orbit: Path, coherence_threshold: float = 0.075):
    """
    Create mask_def.grd based on stacked coherence grids.

    This follows section 8b of the GMTSAR tutorial for unwrapping in regions
    of poor coherence. The process:
    1. Stack coherence grids from all interferograms
    2. Create a mask based on coherence threshold

    Args:
        project_root: Root directory of the project
        orbit: Orbit directory (asc/des)
        coherence_threshold: Minimum coherence value (default 0.075)

    Returns:
        dict: Metadata about the mask creation process
    """
    merge_root = project_root / orbit / "merge"

    # Step 1: Create list of coherence files (Section 8b.i in PDF)
    logger.info("  Listing coherence grids: ls 20[1-9]*/corr.grd > corr.grd_list")
    cmd = "ls 20[1-9]*/corr.grd > corr.grd_list"
    rc, out, err = run_cmd(cmd, cwd=merge_root)

    if rc != 0:
        raise RuntimeError(f"Failed to create corr.grd_list:\n{err}")

    # Step 2: Stack coherence grids (Section 8b.i in PDF)
    logger.info("  Stacking coherence grids: stack.csh corr.grd_list 1 corr_stack.grd std.grd")
    cmd = "stack.csh corr.grd_list 1 corr_stack.grd std.grd"
    rc, out, err = run_cmd(cmd, cwd=merge_root)

    if rc != 0:
        raise RuntimeError(f"stack.csh failed:\n{err}")

    # Step 3: Create mask based on coherence threshold (Section 8b.ii in PDF)
    logger.info(f"  Creating mask: gmt grdmath corr_stack.grd {coherence_threshold} GE 0 NAN = mask_def.grd")
    cmd = f"gmt grdmath corr_stack.grd {coherence_threshold} GE 0 NAN = mask_def.grd"
    rc, out, err = run_cmd(cmd, cwd=merge_root)

    if rc != 0:
        raise RuntimeError(f"gmt grdmath (mask creation) failed:\n{err}")

    logger.info("  mask_def.grd created successfully")


    meta = {
        "coherence_threshold": coherence_threshold,
        "output_files": {
            "corr_stack": "corr_stack.grd",
            "mask_def": "mask_def.grd",
            "std": "std.grd",
        },
        "process": {
            "step1": "Create coherence file list",
            "step2": "Stack coherence grids",
            "step3": f"Create mask with threshold {coherence_threshold}",
        }
    }
    return meta 

def unwrap(project_root: Path, orbit: Path, corr_threshold: float, max_dis_threshold: float, landmask: bool, mask_def: bool = True, num_cores: int = 6):
    merge_dir = project_root / orbit / "merge"

    # Create intflist from merged interferogram directories (only complete ones with phasefilt.grd)
    logger.info("  Creating intflist from merged interferogram directories...")
    intf_dirs = sorted(
        d.name for d in merge_dir.iterdir()
        if d.is_dir() and "_" in d.name and d.name[0:2] == "20"
        and (d / "phasefilt.grd").exists()
    )
    intflist_path = merge_dir / "intflist"
    intflist_path.write_text("\n".join(intf_dirs) + "\n" if intf_dirs else "")

    num_intfs = len(intf_dirs)
    if num_intfs == 0:
        raise RuntimeError("No complete merged interferograms found (no directories with phasefilt.grd)")
    logger.info(f"  intflist created with {num_intfs} interferograms (filtered: only dirs with phasefilt.grd)")

    # Check how many already have unwrap.grd (resume capability)
    already_unwrapped = 0
    if intflist_path.exists():
        for line in intflist_path.read_text().strip().splitlines():
            if line.strip() and (merge_dir / line.strip() / "unwrap.grd").exists():
                already_unwrapped += 1
    if already_unwrapped > 0:
        logger.info(f"  RESUME: {already_unwrapped}/{num_intfs} interferograms already unwrapped, will be skipped")

    # Build unwrap_intf.csh script with resume check (skips if unwrap.grd exists)
    mask_type = "landmask" if landmask else ("mask_def" if mask_def else "none")
    logger.info(f"  Creating unwrap_intf.csh (mask: {mask_type}, corr_threshold: {corr_threshold}, max_dis: {max_dis_threshold})")
    unwrap_script = merge_dir / "unwrap_intf.csh"

    with open(unwrap_script, "w") as f:
        if landmask:
            f.write(f"""#!/bin/csh -f
if (-e $1/unwrap.grd) then
  echo "SKIP: $1/unwrap.grd already exists"
  exit 0
endif
cd $1
ln -sf ../landmask_ra.grd .
snaphu_interp.csh {corr_threshold} {max_dis_threshold}
cd ..
""")
        elif mask_def:
            f.write(f"""#!/bin/csh -f
if (-e $1/unwrap.grd) then
  echo "SKIP: $1/unwrap.grd already exists"
  exit 0
endif
cd $1
ln -sf ../mask_def.grd .
snaphu_interp.csh {corr_threshold} {max_dis_threshold}
cd ..
""")
        else:
            f.write(f"""#!/bin/csh -f
if (-e $1/unwrap.grd) then
  echo "SKIP: $1/unwrap.grd already exists"
  exit 0
endif
cd $1
snaphu_interp.csh {corr_threshold} {max_dis_threshold}
cd ..
""")

    cmd_chmod = f"chmod +x unwrap_intf.csh"
    rc, out, err = run_cmd(cmd_chmod, cwd=merge_dir)
    if rc != 0:
        logger.error(f"  Error making script executable: {err}")
        return

    # Clear unwrap.cmd from previous runs (unwrap_parallel.csh appends to it)
    unwrap_cmd = merge_dir / "unwrap.cmd"
    if unwrap_cmd.exists():
        unwrap_cmd.unlink()

    logger.info(f"  Running unwrap_parallel.csh with {num_cores} cores...")
    logger.info("  This is the most time-consuming step, please be patient...")
    cmd_unwrap = f"unwrap_parallel.csh intflist {num_cores}"
    # Add merge_dir to PATH so unwrap_parallel.csh can find unwrap_intf.csh
    rc, out, err = run_cmd(cmd_unwrap, cwd=merge_dir, extra_path=merge_dir)
    if rc != 0:
        logger.error(f"  unwrap_parallel.csh failed with return code {rc}")
    else:
        logger.info("  unwrap_parallel.csh completed successfully")

    meta = {
        "inputs": {
            "intflist": str(merge_dir / "intflist"),
            "landmask_used": landmask,
            "mask_def_used": mask_def,
        },
        "snaphu_parameters": {
            "correlation threshold": corr_threshold,
            "max discontinuity threshold": max_dis_threshold,
        },
        "execution": {
            "unwrap_script": str(unwrap_script),
            "unwrap_parallel": "unwrap_parallel.csh",
            "command": f"unwrap_parallel.csh intflist {num_cores}",
            "num_cores": num_cores,
            "return_code": rc,
            "stderr": err.strip() if err else None,
        },
    }
    return meta

def write_meta_log(project_root: Path, orbit: str, landmask_info: dict, unwrap_info: dict, mask_def_info: dict):
    meta = {
        "step": 8,
        "orbit": orbit,
        "timestamp": datetime.datetime.now().isoformat(),
        "landmask": landmask_info,
        "unwrap": unwrap_info,  
        "mask_def": mask_def_info
    }   

    logp = (
        project_root
        / "wrapper_meta"
        / "logs"
        / f"step8_{orbit}.json"
    )
    logp.parent.mkdir(parents=True, exist_ok=True)

    with open(logp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return logp


def run_unwrap(project_root: Path, orbit: str, coherence_threshold: float = 0.075,
               corr_threshold: float = 0.01, max_dis_threshold: float = 40,
               use_landmask: bool = False, use_mask_def: bool = True,
               num_cores: int = 6):
    """
    Run complete unwrapping process.

    Args:
        project_root: Root directory of the project
        orbit: Orbit directory name (asc/des)
        coherence_threshold: Threshold for mask_def creation (default 0.075)
        corr_threshold: Correlation threshold for snaphu (default 0.01)
        max_dis_threshold: Max discontinuity threshold for snaphu (default 40)
        use_landmask: Whether to create and use landmask (default False)
        use_mask_def: Whether to create and use mask_def (default True)
        num_cores: Number of parallel cores to use (default 6)

    Returns:
        tuple: (log_path, result_message)
    """
    landmask_info = None
    mask_def_info = None
    result_msg = ""

    # --- STEP 1: Create landmask (if requested) ---
    if use_landmask:
        logger.info("-" * 50)
        logger.info("STEP 1: Creating landmask (landmask.csh)")
        logger.info("-" * 50)
        try:
            landmask_info = make_landmask(project_root, orbit)
            result_msg += f"Landmask created successfully\n"
        except Exception as e:
            result_msg += f"Landmask creation failed: {e}\n"
            logger.error(f"  Landmask creation failed: {e}")

    # --- STEP 2: Create mask_def (if requested) ---
    if use_mask_def:
        logger.info("-" * 50)
        logger.info(f"STEP 2: Creating coherence mask (stack.csh + gmt grdmath, threshold={coherence_threshold})")
        logger.info("-" * 50)
        try:
            mask_def_info = make_mask_def(project_root, orbit, coherence_threshold)
            result_msg += f"Mask_def created with threshold {coherence_threshold}\n"
        except Exception as e:
            result_msg += f"Mask_def creation failed: {e}\n"
            logger.error(f"  Mask_def creation failed: {e}")

    # --- STEP 3: Run unwrapping ---
    logger.info("-" * 50)
    logger.info(f"STEP 3: Unwrapping interferograms (snaphu_interp.csh, {num_cores} cores)")
    logger.info("-" * 50)
    try:
        unwrap_info = unwrap(project_root, orbit, corr_threshold, max_dis_threshold,
                            use_landmask, use_mask_def, num_cores)
        result_msg += f"Unwrapping completed with {num_cores} cores\n"
    except Exception as e:
        result_msg += f"Unwrapping failed: {e}\n"
        unwrap_info = {"error": str(e)}
        logger.error(f"  Unwrapping failed: {e}")

    # Write metadata log
    logp = write_meta_log(project_root, orbit, landmask_info, unwrap_info, mask_def_info)
    result_msg += f"Metadata saved to {logp}\n"

    return logp, result_msg


def main():
    parser = argparse.ArgumentParser(
        description="GMTSAR Step 8: Unwrap Interferograms"
    )
    parser.add_argument("project_root", help="Path to project root directory")
    parser.add_argument("orbit", help="Orbit directory (asc/des)")
    parser.add_argument("--coherence_threshold", type=float, default=0.075,
                       help="Coherence threshold for mask_def (default: 0.075)")
    parser.add_argument("--corr_threshold", type=float, default=0.01,
                       help="Correlation threshold for snaphu (default: 0.01)")
    parser.add_argument("--max_dis_threshold", type=float, default=40,
                       help="Max discontinuity threshold for snaphu (default: 40)")
    parser.add_argument("--use_landmask", action="store_true",
                       help="Create and use landmask for ocean areas")
    parser.add_argument("--use_mask_def", action="store_true", default=True,
                       help="Create and use mask_def for poor coherence (default: True)")

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    orbit = args.orbit

    print(f"Starting unwrapping process for {orbit}...")
    print(f"Project root: {project_root}")
    print(f"Coherence threshold: {args.coherence_threshold}")
    print(f"Correlation threshold: {args.corr_threshold}")
    print(f"Max discontinuity threshold: {args.max_dis_threshold}")
    print(f"Use landmask: {args.use_landmask}")
    print(f"Use mask_def: {args.use_mask_def}")
    print("-" * 60)

    logp, result_msg = run_unwrap(
        project_root,
        orbit,
        coherence_threshold=args.coherence_threshold,
        corr_threshold=args.corr_threshold,
        max_dis_threshold=args.max_dis_threshold,
        use_landmask=args.use_landmask,
        use_mask_def=args.use_mask_def
    )

    print(result_msg)
    print(f"Log file: {logp}")


if __name__ == "__main__":
    main()