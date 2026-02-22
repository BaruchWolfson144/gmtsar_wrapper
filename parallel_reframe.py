#!/usr/bin/env python3
"""
parallel_reframe.py - Parallel reframing for GMTSAR TOPS processing

This module provides parallel reframing capability for Stage 4 of the
GMTSAR wrapper pipeline. It uses directory isolation to run multiple
instances of organize_files_tops_linux.csh simultaneously.

Usage:
    # As module
    from parallel_reframe import run_parallel_reframe
    success, results = run_parallel_reframe(project_root, orbit, num_workers=4)

    # As standalone (for testing)
    python3 parallel_reframe.py /path/to/project asc --workers 4

Author: Claude (with human guidance)
Date: 2026-02-04
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Timeout for single date reframing (seconds)
REFRAME_TIMEOUT = 600  # 10 minutes per date should be plenty

# Temporary directory base (fallback - prefer project-local tmp)
TEMP_BASE_FALLBACK = Path("/tmp")

# GMTSAR script name
REFRAME_SCRIPT = "organize_files_tops_linux.csh"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_safe_dates(safe_filelist: Path) -> dict[str, list[str]]:
    """
    Group SAFE file paths by acquisition date.

    Args:
        safe_filelist: Path to SAFE_filelist file

    Returns:
        Dictionary mapping date strings to list of SAFE paths.
        Example: {'20141009': ['/path/to/S1A...20141009...SAFE', ...]}
    """
    dates_dict = defaultdict(list)

    with open(safe_filelist) as f:
        for line in f:
            safe_path = line.strip()
            if not safe_path:
                continue

            # Extract date from SAFE filename
            # Format: S1A_IW_SLC__1SDV_20141009T155542_...
            match = re.search(r'_(\d{8})T', safe_path)
            if match:
                date = match.group(1)
                dates_dict[date].append(safe_path)
            else:
                print(f"Warning: Could not extract date from {safe_path}")

    return dict(dates_dict)


def validate_work_item(work_item: dict) -> tuple[bool, str]:
    """
    Validate that a work item has all required data.

    Args:
        work_item: Dictionary with date, safe_paths, pins_content, data_dir

    Returns:
        (is_valid, error_message)
    """
    required_keys = ['date', 'safe_paths', 'pins_content', 'data_dir']
    for key in required_keys:
        if key not in work_item:
            return False, f"Missing required key: {key}"

    if not work_item['safe_paths']:
        return False, "No SAFE paths provided"

    if not work_item['pins_content'].strip():
        return False, "Empty pins content"

    # Check that SAFE files exist
    for safe_path in work_item['safe_paths']:
        p = Path(safe_path)
        if not p.exists():
            return False, f"SAFE file not found: {safe_path}"
        if p.is_symlink() and not p.resolve().exists():
            return False, f"Broken symlink: {safe_path}"

    return True, ""


# =============================================================================
# WORKER FUNCTION
# =============================================================================

def reframe_single_date(work_item: dict) -> dict:
    """
    Worker function: reframe a single date in an isolated directory.

    This function is executed in a separate process by ProcessPoolExecutor.
    It creates a temporary workspace, sets up symlinks, runs the GMTSAR
    reframing script, and returns the results.

    Args:
        work_item: Dictionary containing:
            - date: Date string (YYYYMMDD)
            - safe_paths: List of full paths to original SAFE directories
            - pins_content: Content of pins.ll file
            - data_dir: Original data directory path

    Returns:
        Dictionary containing:
            - date: Date string
            - success: Boolean
            - output_dir: Path to F*_F* directory (if successful)
            - safe_count: Number of SAFE files created (if successful)
            - frame_range: Name of F*_F* directory (if successful)
            - duration: Processing time in seconds
            - error: Error message (if failed)
            - stdout: Script stdout (truncated)
            - stderr: Script stderr (truncated)
    """
    date = work_item['date']
    safe_paths = work_item['safe_paths']
    pins_content = work_item['pins_content']

    start_time = time.time()
    result = {
        'date': date,
        'success': False,
        'duration': 0,
    }

    # Validate input
    is_valid, error_msg = validate_work_item(work_item)
    if not is_valid:
        result['error'] = f"Validation failed: {error_msg}"
        result['duration'] = time.time() - start_time
        return result

    # ==========================================================================
    # Step 1: Create isolated temporary directory
    # ==========================================================================
    # Use project-local tmp if provided, otherwise fall back to /tmp
    temp_base_dir = Path(work_item.get('temp_base_dir', str(TEMP_BASE_FALLBACK)))
    # Create temp base directory if it doesn't exist
    temp_base_dir.mkdir(parents=True, exist_ok=True)
    # Include PID to ensure uniqueness even if same date is retried
    temp_base = temp_base_dir / f"reframe_{date}_{os.getpid()}"
    temp_data = temp_base / "data"

    try:
        temp_data.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        result['error'] = f"Failed to create temp directory: {e}"
        result['duration'] = time.time() - start_time
        return result

    try:
        # ======================================================================
        # Step 2: Create symlinks to original SAFE directories
        # ======================================================================
        temp_safe_paths = []
        for orig_path in safe_paths:
            orig_safe = Path(orig_path)
            link_path = temp_data / orig_safe.name

            try:
                link_path.symlink_to(orig_safe.resolve())
                temp_safe_paths.append(str(link_path))
            except FileExistsError:
                # Symlink already exists (shouldn't happen with unique temp dirs)
                temp_safe_paths.append(str(link_path))
            except Exception as e:
                result['error'] = f"Failed to create symlink for {orig_safe.name}: {e}"
                result['duration'] = time.time() - start_time
                return result

        # ======================================================================
        # Step 3: Create SAFE_filelist for this date only
        # ======================================================================
        temp_filelist = temp_data / "SAFE_filelist"
        try:
            temp_filelist.write_text('\n'.join(temp_safe_paths) + '\n')
        except Exception as e:
            result['error'] = f"Failed to write SAFE_filelist: {e}"
            result['duration'] = time.time() - start_time
            return result

        # ======================================================================
        # Step 4: Create pins.ll (same for all dates)
        # ======================================================================
        temp_pins = temp_data / "pins.ll"
        try:
            temp_pins.write_text(pins_content)
        except Exception as e:
            result['error'] = f"Failed to write pins.ll: {e}"
            result['duration'] = time.time() - start_time
            return result

        # ======================================================================
        # Step 5: Change to temp/data directory and run script
        # ======================================================================
        original_dir = os.getcwd()

        try:
            os.chdir(temp_data)

            cmd = f"{REFRAME_SCRIPT} {temp_filelist} {temp_pins} 2"

            proc_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=REFRAME_TIMEOUT
            )

            result['stdout'] = proc_result.stdout[-1000:] if proc_result.stdout else ""
            result['stderr'] = proc_result.stderr[-1000:] if proc_result.stderr else ""

            if proc_result.returncode != 0:
                result['error'] = f"Script returned {proc_result.returncode}"
                result['duration'] = time.time() - start_time
                return result

        except subprocess.TimeoutExpired:
            result['error'] = f"Timeout after {REFRAME_TIMEOUT}s"
            result['duration'] = time.time() - start_time
            return result

        except Exception as e:
            result['error'] = f"Script execution failed: {e}"
            result['duration'] = time.time() - start_time
            return result

        finally:
            os.chdir(original_dir)

        # ======================================================================
        # Step 6: Find output directory
        # ======================================================================
        output_dirs = list(temp_data.glob("F*_F*"))

        if not output_dirs:
            # Check if pins were outside the scene
            if "outside" in result.get('stderr', '').lower():
                result['error'] = "Pins outside scene footprint"
            else:
                result['error'] = "No F*_F* directory created"
            result['duration'] = time.time() - start_time
            return result

        output_dir = output_dirs[0]

        # ======================================================================
        # Step 7: Count created SAFE files
        # ======================================================================
        created_safes = list(output_dir.glob("*.SAFE"))

        if not created_safes:
            result['error'] = "F*_F* directory is empty"
            result['duration'] = time.time() - start_time
            return result

        # ======================================================================
        # Success!
        # ======================================================================
        result['success'] = True
        result['output_dir'] = str(output_dir)
        result['safe_count'] = len(created_safes)
        result['frame_range'] = output_dir.name
        result['duration'] = time.time() - start_time

        return result

    except Exception as e:
        result['error'] = f"Unexpected error: {e}"
        result['duration'] = time.time() - start_time
        return result

    # Note: We don't clean up temp_base here!
    # The main process needs to move the files first.


# =============================================================================
# ORCHESTRATION FUNCTIONS
# =============================================================================

def prepare_work_items(
    project_root: Path,
    orbit: str,
    pins_file: Path = None
) -> tuple[list[dict], str]:
    """
    Prepare work items for parallel reframing.

    Args:
        project_root: Project root directory
        orbit: Orbit type (asc/des)
        pins_file: Optional path to pins.ll file (defaults to {orbit}/data/pins.ll)

    Returns:
        (list of work items, pins_content string)
    """
    data_dir = project_root / orbit / "data"
    safe_filelist = data_dir / "SAFE_filelist"

    # Find pins.ll - check multiple locations
    if pins_file is None:
        possible_pins = [
            data_dir / "pins.ll",
            project_root / orbit / "reframed" / "pins.ll",
        ]
        for p in possible_pins:
            if p.exists():
                pins_file = p
                break
        if pins_file is None:
            raise FileNotFoundError(f"pins.ll not found in any of: {possible_pins}")

    # Read pins.ll
    if not pins_file.exists():
        raise FileNotFoundError(f"pins.ll not found at {pins_file}")
    pins_content = pins_file.read_text()

    # Validate pins content
    lines = [l.strip() for l in pins_content.strip().split('\n') if l.strip()]
    if len(lines) != 2:
        raise ValueError(f"pins.ll must have exactly 2 lines, found {len(lines)}")

    # Read and group SAFE files
    if not safe_filelist.exists():
        raise FileNotFoundError(f"SAFE_filelist not found at {safe_filelist}")

    dates_dict = parse_safe_dates(safe_filelist)

    if not dates_dict:
        raise ValueError("No valid SAFE files found in SAFE_filelist")

    # Create project-local tmp directory for reframing
    project_tmp_dir = project_root / "tmp" / "reframe"
    project_tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Using project-local tmp directory: {project_tmp_dir}")

    # Create work items
    work_items = []
    skipped_single_frame = []

    for date, safe_paths in sorted(dates_dict.items()):
        # Skip dates with only 1 frame (reframing requires 2 frames to stitch)
        if len(safe_paths) < 2:
            skipped_single_frame.append(date)
            continue

        work_items.append({
            'date': date,
            'safe_paths': safe_paths,
            'pins_content': pins_content,
            'data_dir': str(data_dir),
            'temp_base_dir': str(project_tmp_dir),  # Use project-local tmp
        })

    if skipped_single_frame:
        print(f"  Note: Skipping {len(skipped_single_frame)} dates with single frame: {skipped_single_frame[:3]}...")

    return work_items, pins_content


def merge_results(
    project_root: Path,
    orbit: str,
    results: list[dict]
) -> tuple[bool, int, list[str]]:
    """
    Merge all reframed SAFE files into the final F*_F* directory.

    Args:
        project_root: Project root directory
        orbit: Orbit type (asc/des)
        results: List of result dicts from workers

    Returns:
        (success, moved_count, list of errors)
    """
    data_dir = project_root / orbit / "data"
    errors = []

    # Find all unique frame ranges
    frame_ranges = set()
    for r in results:
        if r['success'] and 'frame_range' in r:
            frame_ranges.add(r['frame_range'])

    if not frame_ranges:
        return False, 0, ["No successful reframes to merge"]

    # If multiple frame ranges, warn and use the most common
    if len(frame_ranges) > 1:
        # Count occurrences
        range_counts = defaultdict(int)
        for r in results:
            if r['success'] and 'frame_range' in r:
                range_counts[r['frame_range']] += 1

        target_frame = max(range_counts, key=range_counts.get)
        errors.append(
            f"Multiple frame ranges found: {frame_ranges}. "
            f"Using most common: {target_frame}"
        )
    else:
        target_frame = list(frame_ranges)[0]

    # Create target directory
    target_dir = data_dir / target_frame
    target_dir.mkdir(exist_ok=True)

    # Move all SAFE files
    moved_count = 0
    for r in results:
        if not r['success']:
            continue

        output_dir = Path(r['output_dir'])
        if not output_dir.exists():
            errors.append(f"Output directory missing for {r['date']}: {output_dir}")
            continue

        for safe_dir in output_dir.glob("*.SAFE"):
            dest = target_dir / safe_dir.name

            if dest.exists():
                errors.append(f"Skipped {safe_dir.name}: already exists")
                continue

            try:
                shutil.move(str(safe_dir), str(dest))
                moved_count += 1
            except Exception as e:
                errors.append(f"Failed to move {safe_dir.name}: {e}")

    # Verify final count
    final_count = len(list(target_dir.glob("*.SAFE")))
    expected = sum(r.get('safe_count', 0) for r in results if r['success'])

    if final_count < expected:
        errors.append(f"Count mismatch: expected {expected}, found {final_count}")

    success = (moved_count > 0)
    return success, moved_count, errors


def cleanup_temp_directories(results: list[dict]) -> int:
    """
    Remove all temporary directories created during parallel reframing.

    Args:
        results: List of result dicts from workers

    Returns:
        Number of directories cleaned up
    """
    cleaned = 0

    for r in results:
        if 'output_dir' not in r:
            continue

        output_dir = Path(r['output_dir'])
        temp_base = output_dir.parent.parent  # /tmp/reframe_{date}_{pid}

        if temp_base.exists() and temp_base.name.startswith('reframe_'):
            try:
                shutil.rmtree(temp_base)
                cleaned += 1
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_base}: {e}")

    return cleaned


def run_parallel_reframe(
    project_root: Path,
    orbit: str,
    num_workers: int = 4,
    pins_file: Path = None,
    progress_callback: Optional[callable] = None
) -> tuple[bool, list[dict]]:
    """
    Run reframing in parallel for all dates.

    This is the main entry point for parallel reframing.

    Args:
        project_root: Project root directory
        orbit: Orbit type (asc/des)
        num_workers: Number of parallel workers (default: 4)
        pins_file: Optional path to pins.ll file
        progress_callback: Optional callback(completed, total, result) for progress

    Returns:
        (overall_success, list of result dicts)
    """
    print(f"\n{'='*60}")
    print(f"PARALLEL REFRAMING")
    print(f"{'='*60}")

    # ==========================================================================
    # Phase 1: Preparation
    # ==========================================================================
    print(f"\n[Phase 1] Preparing work items...")

    try:
        work_items, pins_content = prepare_work_items(project_root, orbit, pins_file)
    except Exception as e:
        print(f"ERROR: Preparation failed: {e}")
        return False, []

    total_dates = len(work_items)
    print(f"  Found {total_dates} dates to process")
    print(f"  Using {num_workers} parallel workers")

    # ==========================================================================
    # Phase 2: Parallel Execution
    # ==========================================================================
    print(f"\n[Phase 2] Running parallel reframing...")

    results = []
    completed = 0
    failed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(reframe_single_date, item): item['date']
            for item in work_items
        }

        # Process results as they complete
        for future in as_completed(futures):
            date = futures[future]

            try:
                result = future.result(timeout=REFRAME_TIMEOUT + 60)
                results.append(result)

                completed += 1
                if result['success']:
                    status = f"OK ({result['safe_count']} SAFE, {result['duration']:.1f}s)"
                    symbol = "✓"
                else:
                    failed += 1
                    status = f"FAILED: {result.get('error', 'Unknown')[:40]}"
                    symbol = "✗"

                progress = f"[{completed}/{total_dates}]"
                print(f"  {symbol} {progress} {date}: {status}")

                if progress_callback:
                    progress_callback(completed, total_dates, result)

            except Exception as e:
                failed += 1
                completed += 1
                results.append({
                    'date': date,
                    'success': False,
                    'error': f"Future exception: {e}",
                    'duration': 0
                })
                print(f"  ✗ [{completed}/{total_dates}] {date}: EXCEPTION: {e}")

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Success: {completed - failed}/{total_dates}, Failed: {failed}")

    # ==========================================================================
    # Phase 3: Merge Results
    # ==========================================================================
    print(f"\n[Phase 3] Merging results...")

    merge_success, moved_count, merge_errors = merge_results(
        project_root, orbit, results
    )

    print(f"  Moved {moved_count} SAFE files")
    if merge_errors:
        for err in merge_errors[:5]:  # Show first 5 errors
            print(f"  Warning: {err}")
        if len(merge_errors) > 5:
            print(f"  ... and {len(merge_errors) - 5} more warnings")

    # ==========================================================================
    # Phase 4: Cleanup
    # ==========================================================================
    print(f"\n[Phase 4] Cleaning up temporary directories...")

    cleaned = cleanup_temp_directories(results)
    print(f"  Removed {cleaned} temp directories")

    # ==========================================================================
    # Summary
    # ==========================================================================
    overall_success = (failed == 0 and merge_success)

    print(f"\n{'='*60}")
    if overall_success:
        print(f"PARALLEL REFRAMING COMPLETED SUCCESSFULLY")
    else:
        print(f"PARALLEL REFRAMING COMPLETED WITH ERRORS")
    print(f"{'='*60}\n")

    return overall_success, results


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel reframing for GMTSAR TOPS processing"
    )
    parser.add_argument(
        "project_root",
        help="Project root directory"
    )
    parser.add_argument(
        "orbit",
        choices=["asc", "des"],
        help="Orbit type"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--pins",
        type=str,
        default=None,
        help="Path to pins.ll file (optional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, don't execute"
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    pins_file = Path(args.pins).expanduser().resolve() if args.pins else None

    if args.dry_run:
        print("DRY RUN - showing work items only")
        work_items, pins = prepare_work_items(project_root, args.orbit, pins_file)
        print(f"\nWould process {len(work_items)} dates:")
        for item in work_items[:5]:
            print(f"  {item['date']}: {len(item['safe_paths'])} SAFE files")
        if len(work_items) > 5:
            print(f"  ... and {len(work_items) - 5} more")
        return

    success, results = run_parallel_reframe(
        project_root,
        args.orbit,
        num_workers=args.workers,
        pins_file=pins_file
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
