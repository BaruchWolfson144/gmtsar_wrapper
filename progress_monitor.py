#!/usr/bin/env python3
"""
progress_monitor.py - Progress monitoring for GMTSAR processing stages

This module provides progress tracking for long-running GMTSAR stages:
- Stage 6: Interferogram generation
- Stage 7: Merging subswaths
- Stage 8: SBAS time series (unwrapping)

Usage as standalone:
    python3 progress_monitor.py /path/to/project_root asc --stage 6

Usage in wrapper:
    from progress_monitor import ProgressMonitor
    monitor = ProgressMonitor(project_root, orbit, stage=6)
    monitor.start_monitoring()  # Starts background thread
    # ... run processing ...
    monitor.stop()
"""

import argparse
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta


class ProgressMonitor:
    """Monitor progress of GMTSAR processing stages."""

    def __init__(self, project_root: Path, orbit: str, stage: int, subswath: str = "F1"):
        self.project_root = Path(project_root)
        self.orbit = orbit
        self.stage = stage
        self.subswath = subswath
        self.start_time = None
        self.stop_flag = threading.Event()
        self.monitor_thread = None
        self.last_count = 0
        self.total = 0

    def get_stage6_progress(self, subswath: str = None):
        """Get interferogram generation progress."""
        sub = subswath or self.subswath
        sub_dir = self.project_root / self.orbit / sub

        # Read total expected from intf.in
        intf_in = sub_dir / "intf.in"
        if not intf_in.exists():
            return None, None, "intf.in not found"

        with open(intf_in) as f:
            total = sum(1 for line in f if line.strip())

        # Count completed interferograms (directories with corr.grd)
        intf_dir = sub_dir / "intf"
        if not intf_dir.exists():
            return 0, total, "intf/ directory not created yet"

        # Each completed interferogram has a directory with corr.grd
        completed = 0
        for d in intf_dir.iterdir():
            if d.is_dir() and (d / "corr.grd").exists():
                completed += 1

        return completed, total, None

    def get_stage7_progress(self):
        """Get merge subswaths progress."""
        merge_dir = self.project_root / self.orbit / "merge"
        if not merge_dir.exists():
            return 0, 1, "merge/ not created yet"

        # Check for merged files
        merged_files = list(merge_dir.glob("*.grd"))
        if merged_files:
            return 1, 1, None
        return 0, 1, "Merging in progress"

    def get_stage8_progress(self):
        """Get SBAS progress (unwrapping)."""
        sbas_dir = self.project_root / self.orbit / "SBAS"
        if not sbas_dir.exists():
            return None, None, "SBAS/ not created yet"

        # Count unwrapped files
        unwrap_files = list(sbas_dir.glob("**/unwrap.grd"))
        disp_files = list(sbas_dir.glob("disp_*.grd"))

        # Read expected from intf.in if available
        intf_in = self.project_root / self.orbit / "F1" / "intf.in"
        if intf_in.exists():
            with open(intf_in) as f:
                total = sum(1 for line in f if line.strip())
        else:
            total = max(len(unwrap_files), 1)

        completed = len(unwrap_files)
        return completed, total, None

    def get_progress(self):
        """Get progress for the configured stage."""
        if self.stage == 6:
            return self.get_stage6_progress()
        elif self.stage == 7:
            return self.get_stage7_progress()
        elif self.stage == 8:
            return self.get_stage8_progress()
        else:
            return None, None, f"Unknown stage: {self.stage}"

    def format_time(self, seconds):
        """Format seconds as HH:MM:SS."""
        if seconds < 0:
            return "--:--:--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def format_progress_bar(self, completed, total, width=40):
        """Create a text-based progress bar."""
        if total == 0:
            return "[" + "?" * width + "]"

        pct = completed / total
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def estimate_remaining(self, completed, total):
        """Estimate remaining time based on elapsed time and progress."""
        if self.start_time is None or completed == 0:
            return None

        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = completed / elapsed  # items per second
        remaining = total - completed
        if rate > 0:
            return remaining / rate
        return None

    def print_progress(self, completed, total, error=None, clear_line=True):
        """Print progress to terminal."""
        if error:
            print(f"\r⏳ Stage {self.stage}: {error}", end="", flush=True)
            return

        if total is None or total == 0:
            print(f"\r⏳ Stage {self.stage}: Waiting for data...", end="", flush=True)
            return

        pct = (completed / total) * 100
        bar = self.format_progress_bar(completed, total)

        # Time estimation
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        remaining = self.estimate_remaining(completed, total)
        eta_str = self.format_time(remaining) if remaining else "--:--:--"
        elapsed_str = self.format_time(elapsed)

        # Speed calculation
        if elapsed > 0 and completed > 0:
            speed = completed / elapsed
            speed_str = f"{speed:.2f}/s" if speed >= 1 else f"{speed*60:.1f}/min"
        else:
            speed_str = "---"

        output = (f"\r{bar} {pct:5.1f}% | "
                  f"{completed}/{total} | "
                  f"Elapsed: {elapsed_str} | "
                  f"ETA: {eta_str} | "
                  f"Speed: {speed_str}")

        if clear_line:
            # Clear to end of line
            output += " " * 10

        print(output, end="", flush=True)

    def monitor_loop(self, interval=5):
        """Background monitoring loop."""
        self.start_time = datetime.now()

        while not self.stop_flag.is_set():
            completed, total, error = self.get_progress()

            # Only print if there's a change or first time
            if completed != self.last_count or self.last_count == 0:
                self.print_progress(completed, total, error)
                self.last_count = completed
                self.total = total or 0

            # Check if complete
            if total and completed >= total:
                print()  # New line after completion
                print(f"✅ Stage {self.stage} completed!")
                break

            self.stop_flag.wait(interval)

    def start_monitoring(self, interval=5):
        """Start background monitoring thread."""
        self.monitor_thread = threading.Thread(
            target=self.monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        return self.monitor_thread

    def stop(self):
        """Stop the monitoring thread."""
        self.stop_flag.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def get_summary(self):
        """Get summary of progress."""
        completed, total, error = self.get_progress()
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        return {
            'stage': self.stage,
            'completed': completed,
            'total': total,
            'percentage': (completed / total * 100) if total else 0,
            'elapsed_seconds': elapsed,
            'error': error
        }


def monitor_all_subswaths(project_root: Path, orbit: str, stage: int, interval: int = 5):
    """Monitor progress across all subswaths (F1, F2, F3)."""
    monitors = {}
    for sub in ["F1", "F2", "F3"]:
        monitors[sub] = ProgressMonitor(project_root, orbit, stage, sub)

    start_time = datetime.now()
    print(f"\n📊 Monitoring Stage {stage} progress for all subswaths...")
    print("-" * 70)

    while True:
        all_done = True
        for sub, mon in monitors.items():
            completed, total, error = mon.get_progress()

            if error:
                status = f"⏳ {error}"
            elif total is None:
                status = "⏳ Waiting..."
                all_done = False
            elif completed >= total:
                status = f"✅ Done ({total} interferograms)"
            else:
                pct = completed / total * 100
                bar = mon.format_progress_bar(completed, total, width=20)
                status = f"{bar} {pct:5.1f}% ({completed}/{total})"
                all_done = False

            print(f"\r{sub}: {status}", end="")
            print()  # New line for each subswath

        # Move cursor up for overwrite
        if not all_done:
            print(f"\033[{len(monitors)}A", end="")  # Move up N lines
            time.sleep(interval)
        else:
            break

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Stage {stage} completed in {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description="Monitor GMTSAR processing progress")
    parser.add_argument("project_root", help="Project root directory")
    parser.add_argument("orbit", help="Orbit (asc/des)")
    parser.add_argument("--stage", type=int, default=6, choices=[6, 7, 8],
                        help="Stage to monitor (default: 6)")
    parser.add_argument("--subswath", default="F1", help="Subswath (F1/F2/F3)")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--all-subswaths", action="store_true",
                        help="Monitor all subswaths")

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    if args.all_subswaths:
        monitor_all_subswaths(project_root, args.orbit, args.stage, args.interval)
    else:
        monitor = ProgressMonitor(project_root, args.orbit, args.stage, args.subswath)

        print(f"\n📊 Monitoring Stage {args.stage} ({args.subswath})...")
        print("Press Ctrl+C to stop\n")

        try:
            monitor.monitor_loop(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            summary = monitor.get_summary()
            print(f"Final status: {summary['completed']}/{summary['total']} "
                  f"({summary['percentage']:.1f}%)")


if __name__ == "__main__":
    main()
