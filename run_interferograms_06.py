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
    
def show_intf(dt, db, years, lines, baselines):
    """Display interferogram baseline plot (requires matplotlib)"""
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
    ax.set_title('Baseline vs Time')
    ax.grid(True)


    ax.set_xlim(min(years)-0.5, max(years)+0.5)
    ax.set_ylim(min(baselines)-10, max(baselines)+10)

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    plt.tight_layout()
    plt.savefig("baseline.png", dpi=300)
    plt.show()

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

def copy_and_set_config(project_root: Path, orbit: str, config_path, master: str):
        for sub in ("F1", "F2", "F3"):
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
                if line.strip().startswith("Proc_stage"):
                    line = "Proc_stage = 1"
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
                if line.strip().startswith("threshold_geocode = 0"):
                    line = "threshold_geocode = 0"
                lines.append(line)
            dst.write_text("\n".join(lines))                
        config_info =  {
        "config_source": str(config_path),
        "forced_parameters": {
            "Proc_stage": 1,
            "shift_topo": 0,
            "filter_wavelength": 200,
            "range_dec": 8,
            "azimuth_dec": 2,
            "threshold_snaphu": 0,
            "threshold_geocode": 0,
            },
        }
        return config_info

def make_intf(project_root, orbit, sub="F1", num_cores=6):
    # Run interferogram generation WITHOUT background mode (&)
    # Redirect output to log file properly
    cmd = f"intf_tops_parallel.csh intf.in batch_tops.config {num_cores}"
    cwd = project_root / orbit / sub
    pc, out, err = run_cmd(cmd, cwd)

    # Write log file
    log_file = cwd / "itp.log"
    with open(log_file, "w") as f:
        f.write(f"Command: {cmd}\n")
        f.write(f"Return code: {pc}\n\n")
        f.write(f"Num cores: {num_cores}\n\n")
        f.write("=== STDOUT ===\n")
        f.write(out)
        f.write("\n\n=== STDERR ===\n")
        f.write(err)

    make_intf_info = {
        "command": cmd,
        "subswath": sub,
        "num_cores": num_cores,
        "log_file": str(log_file),
        "return_code": pc,
        "stdout": out.strip()[:500],  # Limit output length
        "stderr": err.strip()[:500],
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

def run_intf(project_root, orbit, sub, threshold_time, threshold_baseline, config_path, master, num_cores=6):
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

    show_intf(threshold_time, threshold_baseline, years, lines, baselines)
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

