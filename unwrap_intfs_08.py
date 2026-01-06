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
    proc = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def make_landmask(project_root: Path, orbit: Path):
    merge_root = project_root / orbit / "merge"

    merge_intf_dir = None
    for d in merge_root.iterdir():
        if d.is_dir() and "_" in d.name:
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

    rc, out, err = run_cmd(
        f"landmask.csh {bounds}",
        cwd=merge_root
    )

    if rc != 0:
        raise RuntimeError(f"landmask failed:\n{err}")

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
    # ls 201*/corr.grd > corr.grd_list
    cmd = "ls 201*/corr.grd > corr.grd_list"
    rc, out, err = run_cmd(cmd, cwd=merge_root)

    if rc != 0:
        raise RuntimeError(f"Failed to create corr.grd_list:\n{err}")

    # Step 2: Stack coherence grids (Section 8b.i in PDF)
    # stack.csh corr.grd_list 1 corr_stack.grd std.grd
    cmd = "stack.csh corr.grd_list 1 corr_stack.grd std.grd"
    rc, out, err = run_cmd(cmd, cwd=merge_root)

    if rc != 0:
        raise RuntimeError(f"stack.csh failed:\n{err}")

    # Step 3: Create mask based on coherence threshold (Section 8b.ii in PDF)
    # gmt grdmath corr_stack.grd 0.075 GE 0 NAN = mask_def.grd
    cmd = f"gmt grdmath corr_stack.grd {coherence_threshold} GE 0 NAN = mask_def.grd"
    rc, out, err = run_cmd(cmd, cwd=merge_root)

    if rc != 0:
        raise RuntimeError(f"gmt grdmath (mask creation) failed:\n{err}")


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

def unwrap(project_root: Path, orbit: Path, corr_threshold: float, max_dis_threshold: float, landmask: bool, mask_def: bool = True):
    merge_dir = project_root / orbit / "merge"
    #make intflist
    cmd = "ls -d 201* > intflist"
    rc, out, err = run_cmd(cmd, cwd=merge_dir)

    #build "unwrap_intf.csh"
    unwrap_script =  merge_dir /"unwrap_intf.csh"
    with open(unwrap_script, "w") as f:
        if landmask:
            f.write(f"""#!/bin/csh -f
            #intflist contains a list of all date1_date2 directories.
            cd $1
            ln -s ../landmask_ra.grd .
            snaphu_interp.csh {corr_threshold} {max_dis_threshold}
            cd ..
            """)
        if mask_def:
            f.write(f"""#!/bin/csh -f
            #intflist contains a list of all date1_date2 directories.
            cd $1
            ln -s ../mask_def.grd .
            snaphu_interp.csh {corr_threshold} {max_dis_threshold}
            cd ..
            """)
        if not landmask and not mask_def:
            f.write(f"""#!/bin/csh -f
            #intflist contains a list of all date1_date2 directories.
            cd $1
            snaphu_interp.csh {corr_threshold} {max_dis_threshold}
            cd ..
            """)
        cmd_chmod = f"chmod +x unwrap_intf.csh"
        rc, out, err = run_cmd(cmd_chmod, cwd=merge_dir)
        if rc != 0:
            print(f"Error making script executable: {err}")
            return
        cmd_unwrap = "unwrap_parallel.csh intflist 6"
        rc, out, err = run_cmd(cmd_unwrap, cwd=merge_dir)
        if rc != 0:
            print(f"Error making script executable: {err}")

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
            "command": "unwrap_parallel.csh intflist 6",
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
        / f"step7_{orbit}.json"
    )
    logp.parent.mkdir(parents=True, exist_ok=True)

    with open(logp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return logp


def run_unwrap(project_root: Path, orbit: str, coherence_threshold: float = 0.075,
               corr_threshold: float = 0.01, max_dis_threshold: float = 40,
               use_landmask: bool = False, use_mask_def: bool = True):
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

    Returns:
        tuple: (log_path, result_message)
    """
    landmask_info = None
    mask_def_info = None
    result_msg = ""

    # Create landmask if requested
    if use_landmask:
        try:
            landmask_info = make_landmask(project_root, orbit)
            result_msg += f"Landmask created successfully\n"
        except Exception as e:
            result_msg += f"Landmask creation failed: {e}\n"

    # Create mask_def if requested
    if use_mask_def:
        try:
            mask_def_info = make_mask_def(project_root, orbit, coherence_threshold)
            result_msg += f"Mask_def created with threshold {coherence_threshold}\n"
        except Exception as e:
            result_msg += f"Mask_def creation failed: {e}\n"

    # Run unwrapping
    try:
        unwrap_info = unwrap(project_root, orbit, corr_threshold, max_dis_threshold,
                            use_landmask, use_mask_def)
        result_msg += f"Unwrapping completed\n"
    except Exception as e:
        result_msg += f"Unwrapping failed: {e}\n"
        unwrap_info = {"error": str(e)}

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