#!/usr/bin/env python3
"""
GMTSAR Pipeline Progress Monitor

Standalone script that monitors stages 5-8 progress by checking output files.
Run from a separate terminal while the pipeline is running.

Usage:
    python monitor.py /path/to/project_root orbit
    python monitor.py /media/oksana/DISK-EXT2/ASC_track58 asc
    python monitor.py /media/oksana/DISK-EXT2/ASC_track58 asc --interval 30
"""

import argparse
import time
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta


# ANSI colors (works in most terminals)
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"


def progress_bar(done, total, width=40):
    if total == 0:
        return f"[{'?' * width}] ?%"
    pct = done / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:6.1%}  ({done}/{total})"


def fmt_eta(seconds):
    if seconds is None or seconds < 0:
        return "calculating..."
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def detect_frame_range(data_dir):
    """Find the F*_F* output directory for stage 5."""
    for d in data_dir.iterdir():
        if d.is_dir() and d.name.startswith("F") and "_F" in d.name:
            return d
    return None


def check_stage5(project_root, orbit):
    """Stage 5: parallel reframe — count .SAFE files in F*_F* vs expected dates."""
    data_dir = project_root / orbit / "data"

    # Total expected: dates with 2+ frames from SAFE_filelist
    safe_filelist = data_dir / "SAFE_filelist"
    if not safe_filelist.exists():
        return None

    dates_dict = {}
    for line in safe_filelist.read_text().strip().splitlines():
        line = line.strip()
        if not line or not line.endswith(".SAFE"):
            continue
        # Extract date: ..._{YYYYMMDD}T...
        basename = Path(line).name
        parts = basename.split("_")
        for p in parts:
            if len(p) >= 15 and p[0] == "2" and "T" in p:
                date = p[:8]
                dates_dict.setdefault(date, []).append(line)
                break

    # Only dates with 2+ frames need reframing
    total_dates = sum(1 for paths in dates_dict.values() if len(paths) >= 2)
    if total_dates == 0:
        return None

    # Count completed: .SAFE files in F*_F* directory
    frame_dir = detect_frame_range(data_dir)
    if frame_dir is None:
        done = 0
    else:
        # Each date produces one .SAFE in the output
        done_dates = set()
        for safe in frame_dir.glob("*.SAFE"):
            parts = safe.name.split("_")
            for p in parts:
                if len(p) >= 15 and p[0] == "2" and "T" in p:
                    done_dates.add(p[:8])
                    break
        done = len(done_dates)

    return {"name": "Stage 5: Reframe", "done": done, "total": total_dates}


def check_stage6(project_root, orbit):
    """Stage 6: interferograms — count completed pairs across subswaths."""
    results = []
    for sw in ["F1", "F2", "F3"]:
        sw_dir = project_root / orbit / sw
        intf_in = sw_dir / "intf.in"
        intf_all = sw_dir / "intf_all"

        if not intf_in.exists():
            continue

        # Total pairs from intf.in
        total = sum(1 for l in intf_in.read_text().strip().splitlines() if l.strip() and ":" in l)

        # Completed: dirs in intf_all with phasefilt.grd + corr.grd
        done = 0
        if intf_all.exists():
            for d in intf_all.iterdir():
                if d.is_dir() and "_" in d.name:
                    if (d / "phasefilt.grd").exists() and (d / "corr.grd").exists():
                        done += 1

        results.append({"name": f"  {sw}", "done": done, "total": total})

    if not results:
        return None

    # Also compute aggregate
    total_all = sum(r["total"] for r in results)
    done_all = sum(r["done"] for r in results)
    return {"name": "Stage 6: Interferograms", "done": done_all, "total": total_all, "sub": results}


def check_stage7(project_root, orbit):
    """Stage 7: merge — count merged interferograms with phasefilt.grd."""
    merge_dir = project_root / orbit / "merge"
    if not merge_dir.exists():
        return None

    # Total: from intflist or merge_list
    intflist = merge_dir / "intflist"
    if not intflist.exists():
        return None

    total = sum(1 for l in intflist.read_text().strip().splitlines() if l.strip())
    if total == 0:
        return None

    # Done: directories with phasefilt.grd
    done = 0
    for d in merge_dir.iterdir():
        if d.is_dir() and "_" in d.name and d.name[0].isdigit():
            if (d / "phasefilt.grd").exists():
                done += 1

    return {"name": "Stage 7: Merge", "done": done, "total": total}


def check_stage8(project_root, orbit):
    """Stage 8: unwrap — count unwrapped interferograms."""
    merge_dir = project_root / orbit / "merge"
    if not merge_dir.exists():
        return None

    # Total: from intflist in merge/
    intflist = merge_dir / "intflist"
    if not intflist.exists():
        return None

    total = sum(1 for l in intflist.read_text().strip().splitlines() if l.strip())
    if total == 0:
        return None

    # Done: directories with unwrap.grd
    done = 0
    for d in merge_dir.iterdir():
        if d.is_dir() and "_" in d.name and d.name[0].isdigit():
            if (d / "unwrap.grd").exists():
                done += 1

    return {"name": "Stage 8: Unwrap", "done": done, "total": total}


def detect_active_stage(stages):
    """Find which stage is currently running (has progress but not complete)."""
    for stage in stages:
        if stage is None:
            continue
        if stage["done"] < stage["total"] and stage["done"] > 0:
            return stage
    # If no stage has partial progress, find the first with 0 done but total > 0
    for stage in stages:
        if stage is None:
            continue
        if stage["done"] == 0 and stage["total"] > 0:
            return stage
    return None


class EtaTracker:
    """Tracks progress rate and estimates time remaining."""

    def __init__(self):
        self.history = {}  # stage_name -> [(timestamp, done_count)]

    def update(self, name, done):
        now = time.time()
        if name not in self.history:
            self.history[name] = []
        hist = self.history[name]
        # Only record if count changed or first sample
        if not hist or hist[-1][1] != done:
            hist.append((now, done))
        # Keep last 20 samples to smooth rate
        if len(hist) > 20:
            hist.pop(0)

    def get_eta(self, name, done, total):
        remaining = total - done
        if remaining <= 0:
            return 0
        hist = self.history.get(name, [])
        if len(hist) < 2:
            return None
        # Rate from oldest sample to newest
        t0, d0 = hist[0]
        t1, d1 = hist[-1]
        dt = t1 - t0
        dd = d1 - d0
        if dd <= 0 or dt <= 0:
            return None
        rate = dd / dt  # items per second
        return remaining / rate


def main():
    parser = argparse.ArgumentParser(description="GMTSAR Pipeline Progress Monitor")
    parser.add_argument("project_root", help="Path to project root directory")
    parser.add_argument("orbit", help="Orbit directory (asc/des)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Refresh interval in seconds (default: 15)")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    orbit = args.orbit
    interval = args.interval

    if not project_root.exists():
        print(f"Error: {project_root} does not exist")
        sys.exit(1)

    eta_tracker = EtaTracker()

    print(f"{BOLD}GMTSAR Pipeline Monitor{RESET}")
    print(f"Project: {project_root}")
    print(f"Orbit:   {orbit}")
    print(f"Refresh: every {interval}s")
    print(f"Press Ctrl+C to exit\n")

    try:
        while True:
            # Collect stage info
            stages = [
                check_stage5(project_root, orbit),
                check_stage6(project_root, orbit),
                check_stage7(project_root, orbit),
                check_stage8(project_root, orbit),
            ]

            active = detect_active_stage(stages)

            # Clear screen and redraw
            os.system("clear" if os.name != "nt" else "cls")

            now_str = datetime.now().strftime("%H:%M:%S")
            print(f"{BOLD}GMTSAR Pipeline Monitor{RESET}  [{now_str}]")
            print(f"{DIM}Project: {project_root} / {orbit}{RESET}")
            print()

            for stage in stages:
                if stage is None:
                    continue

                is_active = (active is not None and stage["name"] == active["name"])
                done = stage["done"]
                total = stage["total"]

                # Update ETA tracker
                eta_tracker.update(stage["name"], done)
                eta = eta_tracker.get_eta(stage["name"], done, total)

                # Status indicator
                if done >= total:
                    status = f"{GREEN}DONE{RESET}"
                    marker = f"{GREEN}✓{RESET}"
                elif is_active:
                    status = f"{YELLOW}RUNNING{RESET}"
                    marker = f"{YELLOW}▶{RESET}"
                elif done > 0:
                    status = f"{CYAN}PARTIAL{RESET}"
                    marker = f"{CYAN}~{RESET}"
                else:
                    status = f"{DIM}WAITING{RESET}"
                    marker = f"{DIM}·{RESET}"

                print(f" {marker} {BOLD}{stage['name']}{RESET}  {status}")
                print(f"   {progress_bar(done, total)}")

                if is_active and eta is not None:
                    print(f"   ETA: ~{fmt_eta(eta)}")
                elif done > 0 and done < total:
                    eta_str = fmt_eta(eta) if eta else "calculating..."
                    print(f"   ETA: ~{eta_str}")

                # Sub-items (e.g., per-subswath for stage 6)
                if "sub" in stage:
                    for sub in stage["sub"]:
                        sub_done = sub["done"]
                        sub_total = sub["total"]
                        if sub_total > 0:
                            pct = sub_done / sub_total * 100
                            print(f"   {DIM}{sub['name']}: {sub_done}/{sub_total} ({pct:.0f}%){RESET}")

                print()

            # Footer
            print(f"{DIM}Refreshing every {interval}s  |  Ctrl+C to exit{RESET}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor stopped.{RESET}")


if __name__ == "__main__":
    main()
