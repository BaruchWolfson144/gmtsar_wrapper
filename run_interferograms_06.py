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

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def run_cmd(cmd, cwd=None):
    import os
    # Preserve the full environment including PATH
    env = os.environ.copy()
    proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def preparing_intf_list(project_root: Path, orbit: str, sub: str, threshold_time: int, threshold_baseline: int):
    #cmd = f"select_pairs.csh baseline_table.dat {threshold_time} {threshold_baseline}"
    #run_cmd(cmd=cmd, cwd=subswath)

    # re-write select_pairs.csh unstable script
    baseline_table = project_root / orbit / sub / "raw" / "baseline_table.dat"
    intf_in_path = project_root / orbit / sub / "intf.in"
    dt = float(threshold_time)
    db = float(threshold_baseline)

    lines = []
    years = []
    baselines = []

    # Check if baseline_table.dat exists
    if not baseline_table.exists():
        raise FileNotFoundError(
            f"baseline_table.dat not found at {baseline_table}\n"
            f"This file should have been created during Stage 05 (preprocessing).\n"
            f"Please ensure Stage 05 completed successfully before running Stage 06.\n"
            f"Check that you're using the correct project_root directory."
        )

    with open(baseline_table) as f:
        for line in f:
            if line.strip():
                parts = line.split()
                name = parts[0]
                t = float(parts[2])
                b = float(parts[4])
                lines.append((name, t, b))
                years.append(2014 + t / 365.25)
                baselines.append(b)

    with open(intf_in_path, "w") as fout:
        num_intf = 0
        for i, (n1, t1, b1) in enumerate(lines):
            for j, (n2, t2, b2) in enumerate(lines):
                if t1 < t2:
                    dt12 = t2 - t1
                    db12 = abs(b2 - b1)
                    if dt12 < dt and db12 < db:
                        fout.write(f"{n1}:{n2}\n")
                        num_intf +=1
    
    intf_list_info = {
        "intf.in": "created",
        "subswath": sub,
        "threshold_time_days": threshold_time,
        "threshold_baseline_m": threshold_baseline,
        "num_interferograms": num_intf,

    }
    return lines, years, baselines, intf_list_info
    
def show_intf(dt, db, years, lines, baselines, output_path=None, title_suffix=""):
    """Display interferogram baseline plot (requires matplotlib).

    Args:
        dt: Temporal threshold (days)
        db: Baseline threshold (meters)
        years: List of years for each scene
        lines: List of (name, time, baseline) tuples
        baselines: List of baseline values
        output_path: Path to save the plot. If None, saves to 'baseline.png' in current dir
        title_suffix: Optional suffix for the plot title (e.g., " - F1")
    """
    if not HAS_MATPLOTLIB:
        print("Note: matplotlib not available, skipping baseline plot")
        return

    # Check if we have data to plot
    if not years or not baselines:
        print("Note: No baseline data available for plotting")
        return

    # graph plotting
    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(years, baselines, s=20, c='black', label='Baseline Points')

    for n1, t1, b1 in lines:
        for n2, t2, b2 in lines:
            if t1 < t2:
                dt12 = t2 - t1
                db12 = abs(b2 - b1)
                if dt12 < dt and db12 < db:
                    y1 = 2014 + t1 / 365.25
                    y2 = 2014 + t2 / 365.25
                    ax.plot([y1, y2], [b1, b2], 'r-', alpha=0.5)

    ax.set_xlabel('Year')
    ax.set_ylabel('Baseline (m)')
    ax.set_title(f'Baseline vs Time{title_suffix}')
    ax.grid(True)


    ax.set_xlim(min(years)-0.5, max(years)+0.5)
    ax.set_ylim(min(baselines)-10, max(baselines)+10)

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    plt.tight_layout()
    save_path = output_path if output_path else "baseline.png"
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close figure to free memory - don't use plt.show() in subprocess
    print(f"Baseline plot saved to: {save_path}")

def copy_intf(project_root: Path, orbit: str):
    intf_in = project_root / orbit / "F1" / "intf.in"  
    for sub in ("F2", "F3"):
        dest= project_root / orbit / sub / "intf.in"
        shutil.copy2(intf_in, dest)
        text = dest.read_text()
        text = text.replace("F1", sub)
        dest.write_text(text)

    copy_intf_info = {
        "file": "intf.in",
        "source": "F1",
        "copid": "F2, F3"
    }
    return copy_intf_info

def copy_and_set_config(project_root: Path, orbit: str, config_path, master: str, subswath_list=None):
        """Copy and configure batch_tops.config for specified subswaths.

        Args:
            subswath_list: List of subswaths to process. If None, processes all (F1, F2, F3).
        """
        if subswath_list is None:
            subswath_list = ["F1", "F2", "F3"]

        for sub in subswath_list:
            dst = project_root / orbit / sub / "batch_tops.config"
            # Skip if source and destination are the same file
            if Path(config_path).resolve() != dst.resolve():
                shutil.copy2(config_path, dst)
            lines = []
            # Create subswath-specific master name without modifying original
            master_for_sub = master[:-2] + sub
            for line in dst.read_text().splitlines():
                if line.strip().startswith("master_image"):
                    line = f"master_image = {master_for_sub}"
                # NOTE: Proc_stage is now handled dynamically by make_intf()
                # Phase 1 runs with stage=1, Phase 2 runs with stage=2
                if line.strip().startswith("shift_topo"):
                    line = "shift_topo = 0"
                if line.strip().startswith("filter_wavelength"):
                    line = "filter_wavelength = 200"
                if line.strip().startswith("range_dec"):
                    line = "range_dec = 8"
                if line.strip().startswith("azimuth_dec"):
                    line = "azimuth_dec = 2"
                if line.strip().startswith("threshold_snaphu"):
                    line = "threshold_snaphu = 0"
                if line.strip().startswith("threshold_geocode"):
                    line = "threshold_geocode = 0"
                lines.append(line)
            dst.write_text("\n".join(lines))
        config_info =  {
        "config_source": str(config_path),
        "subswaths_configured": subswath_list,
        "forced_parameters": {
            "Proc_stage": "dynamic (1 for first intf, 2 for rest)",
            "shift_topo": 0,
            "filter_wavelength": 200,
            "range_dec": 8,
            "azimuth_dec": 2,
            "threshold_snaphu": 0,
            "threshold_geocode": 0,
            },
        }
        return config_info

def update_proc_stage(config_path: Path, stage: int):
    """Update Proc_stage in batch_tops.config file."""
    lines = []
    for line in config_path.read_text().splitlines():
        if line.strip().startswith("proc_stage") or line.strip().startswith("Proc_stage"):
            line = f"proc_stage = {stage}"
        lines.append(line)
    config_path.write_text("\n".join(lines))


def get_completed_intfs(cwd: Path) -> set:
    """
    Detect completed interferograms by scanning intf_all/ directory.

    Completed interferograms are stored in directories like:
        intf_all/2019123_2020015/

    Each directory should contain key output files (phasefilt.grd, corr.grd)
    to be considered complete.

    Returns:
        Set of completed pair strings in format "scene1:scene2"
        (matching intf.in format)
    """
    intf_all = cwd / "intf_all"
    completed = set()

    if not intf_all.exists():
        return completed

    for d in intf_all.iterdir():
        if not d.is_dir():
            continue

        # Directory name format: YYYYDOY_YYYYDOY (e.g., 2019123_2020015)
        name = d.name
        if "_" not in name:
            continue

        # Check for key output files that indicate completion
        # phasefilt.grd is created at the end of interferogram processing
        phasefilt = d / "phasefilt.grd"
        corr = d / "corr.grd"

        if phasefilt.exists() and corr.exists():
            # Convert directory name to intf.in format
            # Directory: 2019123_2020015 → intf.in: S1_20190503_...:S1_20200115_...
            # We need to find the actual scene names from the PRM files in the directory
            prm_files = list(d.glob("*.PRM"))
            if len(prm_files) >= 2:
                # Extract scene stems from PRM files
                scenes = sorted([p.stem for p in prm_files if p.stem.startswith("S1_")])
                if len(scenes) >= 2:
                    # Format: scene1:scene2
                    pair_str = f"{scenes[0]}:{scenes[1]}"
                    completed.add(pair_str)

    return completed


def parse_intf_pair(pair_str: str) -> tuple:
    """
    Parse an interferogram pair string into (ref_scene, rep_scene).

    Input format: "S1_20191206_ALL_F1:S1_20200104_ALL_F1"
    Returns: ("S1_20191206_ALL_F1", "S1_20200104_ALL_F1")
    """
    parts = pair_str.strip().split(":")
    if len(parts) == 2:
        return (parts[0], parts[1])
    return None


def make_intf(project_root, orbit, sub="F1", num_cores=6):
    """
    Run interferogram generation using the two-phase approach from the GMTSAR manual:

    PHASE 1: Run ONE interferogram with Proc_stage = 1 to create topo_ra.grd
    PHASE 2: Run ALL REMAINING interferograms with Proc_stage = 2 (uses existing topo_ra.grd)

    This prevents the race condition where multiple parallel jobs try to create topo_ra.grd
    simultaneously, causing file conflicts and broken symlinks.

    RESUME CAPABILITY: If topo_ra.grd already exists, skips Phase 1.
    If interferograms already exist in intf_all/, skips those pairs in Phase 2.
    """
    cwd = project_root / orbit / sub
    config_file = cwd / "batch_tops.config"
    intf_in = cwd / "intf.in"
    log_file = cwd / "itp.log"
    topo_ra = cwd / "topo" / "topo_ra.grd"

    # Read all interferogram pairs
    with open(intf_in) as f:
        all_pairs = [line.strip() for line in f if line.strip()]

    if not all_pairs:
        return {
            "command": "none",
            "subswath": sub,
            "num_cores": num_cores,
            "error": "No interferogram pairs found in intf.in",
            "return_code": 1,
        }

    results = {"phase1": {}, "phase2": {}}

    # ========================================
    # RESUME: Check for completed interferograms
    # ========================================
    completed_intfs = get_completed_intfs(cwd)
    if completed_intfs:
        print(f"[{sub}] RESUME: Found {len(completed_intfs)} completed interferograms in intf_all/")

    # ========================================
    # PHASE 1: Create topo_ra.grd with ONE interferogram
    # ========================================
    # RESUME: Skip Phase 1 if topo_ra.grd already exists
    if topo_ra.exists():
        print(f"[{sub}] PHASE 1 SKIPPED: topo_ra.grd already exists (resume mode)")
        results["phase1"] = {
            "skipped": True,
            "reason": "topo_ra.grd already exists (resume mode)",
            "topo_ra_path": str(topo_ra),
        }
        pc1 = 0
        # For Phase 2, we start with all_pairs since Phase 1 was skipped
        phase1_pair = None
    else:
        print(f"[{sub}] PHASE 1: Running first interferogram with Proc_stage=1 to create topo_ra.grd")

        # Ensure Proc_stage = 1
        update_proc_stage(config_file, 1)

        # Find first pair that isn't already completed
        phase1_pair = None
        for pair in all_pairs:
            if pair not in completed_intfs:
                phase1_pair = pair
                break

        if phase1_pair is None:
            # All pairs are already completed - nothing to do
            print(f"[{sub}] All {len(all_pairs)} interferograms already completed!")
            return {
                "command": "none",
                "subswath": sub,
                "num_cores": num_cores,
                "total_pairs": len(all_pairs),
                "skipped_pairs": len(completed_intfs),
                "message": "All interferograms already completed (resume mode)",
                "return_code": 0,
            }

        # Create one.in with just the first incomplete pair
        one_in = cwd / "one.in"
        with open(one_in, "w") as f:
            f.write(phase1_pair + "\n")

        # Run single interferogram to create topo_ra.grd
        cmd1 = "intf_tops.csh one.in batch_tops.config"
        pc1, out1, err1 = run_cmd(cmd1, cwd)

        results["phase1"] = {
            "command": cmd1,
            "pair": phase1_pair,
            "return_code": pc1,
            "stdout": out1.strip()[:500],
            "stderr": err1.strip()[:500],
        }

        # Check if topo_ra.grd was created
        if not topo_ra.exists():
            error_msg = f"PHASE 1 FAILED: topo_ra.grd was not created. Check logs."
            print(f"[{sub}] ERROR: {error_msg}")
            with open(log_file, "w") as f:
                f.write(f"PHASE 1 ERROR: {error_msg}\n")
                f.write(f"Command: {cmd1}\n")
                f.write(f"Return code: {pc1}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(out1)
                f.write("\n\n=== STDERR ===\n")
                f.write(err1)
            return {
                "command": cmd1,
                "subswath": sub,
                "num_cores": num_cores,
                "error": error_msg,
                "return_code": pc1,
                "phase1_results": results["phase1"],
            }

        print(f"[{sub}] PHASE 1 SUCCESS: topo_ra.grd created")

    # ========================================
    # PHASE 2: Run remaining interferograms with Proc_stage = 2
    # ========================================
    # Build list of pairs to process (excluding Phase 1 pair and completed ones)
    remaining_pairs = []
    skipped_count = 0
    for pair in all_pairs:
        if phase1_pair and pair == phase1_pair:
            continue  # Skip the Phase 1 pair
        if pair in completed_intfs:
            skipped_count += 1
            continue  # Skip already completed pairs
        remaining_pairs.append(pair)

    if skipped_count > 0:
        print(f"[{sub}] RESUME: Skipping {skipped_count} already completed interferograms")

    if remaining_pairs:
        print(f"[{sub}] PHASE 2: Running {len(remaining_pairs)} remaining interferograms with Proc_stage=2")

        # Change to Proc_stage = 2 (skips topo_ra.grd creation)
        update_proc_stage(config_file, 2)

        # Create remaining.in with all remaining pairs
        remaining_in = cwd / "remaining.in"
        with open(remaining_in, "w") as f:
            for pair in remaining_pairs:
                f.write(pair + "\n")

        # Run remaining interferograms in parallel
        cmd2 = f"intf_tops_parallel.csh remaining.in batch_tops.config {num_cores}"
        pc2, out2, err2 = run_cmd(cmd2, cwd)

        results["phase2"] = {
            "command": cmd2,
            "num_pairs": len(remaining_pairs),
            "return_code": pc2,
            "stdout": out2.strip()[:500],
            "stderr": err2.strip()[:500],
        }

        print(f"[{sub}] PHASE 2 completed with return code {pc2}")
    else:
        if skipped_count > 0 and not remaining_pairs:
            # All pairs were already completed (except possibly Phase 1)
            print(f"[{sub}] PHASE 2 SKIPPED: All remaining interferograms already completed")
            results["phase2"] = {"skipped": True, "reason": f"All remaining interferograms already completed (resume mode)", "skipped_count": skipped_count}
        else:
            print(f"[{sub}] PHASE 2 SKIPPED: Only one interferogram pair total")
            results["phase2"] = {"skipped": True, "reason": "Only one interferogram pair"}
        pc2 = 0

    # Write comprehensive log file
    with open(log_file, "w") as f:
        f.write(f"=== INTERFEROGRAM GENERATION LOG ===\n")
        f.write(f"Subswath: {sub}\n")
        f.write(f"Total pairs: {len(all_pairs)}\n")
        f.write(f"Num cores: {num_cores}\n")
        f.write(f"Previously completed: {len(completed_intfs)}\n\n")

        f.write(f"=== PHASE 1: Create topo_ra.grd ===\n")
        if results["phase1"].get("skipped"):
            f.write(f"SKIPPED: {results['phase1'].get('reason', 'unknown')}\n")
        else:
            f.write(f"Command: {results['phase1'].get('command', 'N/A')}\n")
            f.write(f"Pair: {results['phase1'].get('pair', 'N/A')}\n")
            f.write(f"Return code: {pc1}\n")
            f.write(f"STDOUT:\n{results['phase1'].get('stdout', '')}\n")
            f.write(f"STDERR:\n{results['phase1'].get('stderr', '')}\n\n")

        f.write(f"=== PHASE 2: Remaining interferograms ===\n")
        if results["phase2"].get("skipped"):
            f.write(f"SKIPPED: {results['phase2'].get('reason', 'unknown')}\n")
        else:
            f.write(f"Command: {results['phase2'].get('command', 'N/A')}\n")
            f.write(f"Num pairs processed: {len(remaining_pairs)}\n")
            f.write(f"Skipped (already completed): {skipped_count}\n")
            f.write(f"Return code: {pc2}\n")
            f.write(f"STDOUT:\n{results['phase2'].get('stdout', '')}\n")
            f.write(f"STDERR:\n{results['phase2'].get('stderr', '')}\n")

    make_intf_info = {
        "command": "two-phase with resume",
        "subswath": sub,
        "num_cores": num_cores,
        "log_file": str(log_file),
        "return_code": max(pc1, pc2),
        "total_pairs": len(all_pairs),
        "skipped_pairs": skipped_count + (1 if results["phase1"].get("skipped") else 0),
        "processed_pairs": len(remaining_pairs) + (0 if results["phase1"].get("skipped") else 1),
        "phase1": results["phase1"],
        "phase2": results["phase2"],
    }
    return make_intf_info

def write_meta_log(project_root: Path, orbit: str, subswath: str, intf_list_info, copy_intf_info, config_info, make_intf_info):
    meta = {
        "step": 6,
        "orbit": orbit,
        "subswath": subswath,
        "timestamp": datetime.datetime.now().isoformat(),
        "preparing_intf_list": intf_list_info,
        "copy_intf":copy_intf_info,
        "copy_and_set_config":config_info,
        "make_intf":make_intf_info
    }   
    logp = (
        project_root
        / "wrapper_meta"
        / "logs"
        / f"step6_{orbit}_{subswath}.json"
    )
    logp.parent.mkdir(parents=True, exist_ok=True)

    with open(logp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return logp

def run_intf_single(project_root, orbit, sub, threshold_time, threshold_baseline, config_path, master, num_cores=6):
    """Run interferogram generation for a single subswath.

    This function is safe for parallel execution - it only operates on the specified subswath.
    Prerequisites: batch_tops.config must already be copied to the subswath directory.
    """
    lines, years, baselines, intf_list_info = preparing_intf_list(project_root, orbit, sub, threshold_time, threshold_baseline)

    # Check if any interferograms were created
    num_intf = intf_list_info.get("num_interferograms", 0)
    if num_intf == 0:
        print(f"WARNING: No interferogram pairs found for subswath {sub}!")
        print(f"  Threshold time: {threshold_time} days")
        print(f"  Threshold baseline: {threshold_baseline} meters")
        print(f"  Number of scenes: {len(lines)}")
        if len(lines) < 2:
            print(f"  ERROR: Need at least 2 scenes to create interferograms. Only {len(lines)} scene(s) available.")
        print("  Consider:")
        print("    - Increasing threshold_time and/or threshold_baseline")
        print("    - Downloading more scenes from different dates")

    # Save baseline plot to subswath directory (unique path for parallel execution)
    plot_path = project_root / orbit / sub / f"baseline_{sub}.png"
    show_intf(threshold_time, threshold_baseline, years, lines, baselines,
              output_path=str(plot_path), title_suffix=f" - {sub}")

    # Note: copy_intf is not needed - each subswath generates its own intf.in
    # Note: copy_and_set_config should be called BEFORE parallel execution

    make_intf_info = make_intf(project_root, orbit, sub=sub, num_cores=num_cores)

    # Minimal log info for parallel execution
    copy_intf_info = {"skipped": True, "reason": "parallel mode - each subswath has its own intf.in"}
    config_info = {"skipped": True, "reason": "parallel mode - config copied before parallel execution"}

    return write_meta_log(project_root, orbit, sub, intf_list_info, copy_intf_info, config_info, make_intf_info)


def run_intf(project_root, orbit, sub, threshold_time, threshold_baseline, config_path, master, num_cores=6):
    """Run interferogram generation for a single subswath (legacy sequential mode).

    Note: This function copies config to ALL subswaths, so it's not safe for parallel execution.
    Use run_intf_single() for parallel execution after preparing configs with copy_and_set_config().
    """
    lines, years, baselines, intf_list_info = preparing_intf_list(project_root, orbit, sub, threshold_time, threshold_baseline)

    # Check if any interferograms were created
    num_intf = intf_list_info.get("num_interferograms", 0)
    if num_intf == 0:
        print(f"WARNING: No interferogram pairs found for subswath {sub}!")
        print(f"  Threshold time: {threshold_time} days")
        print(f"  Threshold baseline: {threshold_baseline} meters")
        print(f"  Number of scenes: {len(lines)}")
        if len(lines) < 2:
            print(f"  ERROR: Need at least 2 scenes to create interferograms. Only {len(lines)} scene(s) available.")
        print("  Consider:")
        print("    - Increasing threshold_time and/or threshold_baseline")
        print("    - Downloading more scenes from different dates")

    # Save baseline plot to subswath directory
    plot_path = project_root / orbit / sub / f"baseline_{sub}.png"
    show_intf(threshold_time, threshold_baseline, years, lines, baselines,
              output_path=str(plot_path), title_suffix=f" - {sub}")
    copy_intf_info = copy_intf(project_root, orbit)
    config_info = copy_and_set_config(project_root, orbit, config_path, master)
    make_intf_info = make_intf(project_root, orbit, sub=sub, num_cores=num_cores)
    return(write_meta_log(project_root, orbit, sub, intf_list_info, copy_intf_info, config_info, make_intf_info))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=str, help="path to project root")
    parser.add_argument("orbit", type=str, default="asc", help="asc or dec")
    parser.add_argument("subswath", type=str, default="F1", help="subswath to work on (F1/F2/F3)")
    parser.add_argument("threshold_time", type=int, help="threshold time for choosing interferograms")
    parser.add_argument("threshold_baseline", type=int, help="threshold baseline for choosing interferograms")
    parser.add_argument("config_path", type=str, help="path to config_bach")
    parser.add_argument("master", type=str, help="full name of master file")
    
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    orbit = args.orbit
    sub = args.subswath
    threshold_time = args.threshold_time
    threshold_baseline = args.threshold_baseline
    config_path = Path(args.config_path).expanduser().resolve()
    master = args.master

    run_intf(project_root, orbit, sub, threshold_time, threshold_baseline, config_path, master)

if __name__ == "__main__":
    main()

