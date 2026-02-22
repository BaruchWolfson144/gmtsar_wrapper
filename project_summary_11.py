#!/usr/bin/env python3
"""
================================================================================
GMTSAR Stage 11: Project Summary Report
================================================================================

Generates a comprehensive text summary of the entire InSAR processing pipeline.
Collects metadata from:
- Config file (YAML)
- Project state (JSON)
- Stage log files (JSON)
- Direct file/directory counting on disk
- Cross-validation between logs and actual files

Output: {project_root}/wrapper_meta/project_summary.txt

STATUS: Production - integrated into main.py
Author: Claude Code Assistant
Date: 2026-02-22
================================================================================
"""

import argparse
import json
import os
import logging
from pathlib import Path
import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_yaml_config(config_path: Path) -> Dict:
    """Load YAML config, falling back to simple parser if PyYAML unavailable."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # Simple YAML-like parser for flat keys
        config = {}
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    key, _, val = line.partition(':')
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if val.lower() == 'true':
                        val = True
                    elif val.lower() == 'false':
                        val = False
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                    config[key] = val
        return config


def load_json(path: Path) -> Optional[Dict]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def count_files(directory: Path, pattern: str) -> int:
    """Count files matching a glob pattern in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def count_dirs(directory: Path, prefix: str = "20") -> int:
    """Count directories starting with a prefix (date-based dirs)."""
    if not directory.exists():
        return 0
    return sum(1 for d in directory.iterdir()
               if d.is_dir() and d.name.startswith(prefix))


def count_dirs_with_file(directory: Path, filename: str, prefix: str = "20") -> int:
    """Count date-directories containing a specific file."""
    if not directory.exists():
        return 0
    return sum(1 for d in directory.iterdir()
               if d.is_dir() and d.name.startswith(prefix) and (d / filename).exists())


def fmt_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def dir_size(path: Path) -> int:
    """Calculate total size of a directory (non-recursive for symlinks)."""
    total = 0
    if not path.exists():
        return 0
    try:
        for f in path.rglob("*"):
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def parse_baseline_table(path: Path) -> List[Dict]:
    """Parse baseline_table.dat and return list of scene dicts."""
    scenes = []
    if not path.exists():
        return scenes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                scenes.append({
                    "stem": parts[0],
                    "decimal_year": float(parts[1]),
                    "day": int(parts[2]),
                    "parallel_bl": float(parts[3]),
                    "perpendicular_bl": float(parts[4]),
                })
    return scenes


def parse_intf_in(path: Path) -> List[Tuple[str, str]]:
    """Parse intf.in and return list of (ref, sec) tuples."""
    pairs = []
    if not path.exists():
        return pairs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
    return pairs


def scene_stem_to_date(stem: str) -> Optional[str]:
    """Extract YYYYMMDD date from scene stem like S1_20191206_ALL_F1."""
    parts = stem.split('_')
    if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[1]
    return None


# =============================================================================
# SECTION BUILDERS
# =============================================================================

def build_header(project_root: Path, orbit: str, config: Dict) -> List[str]:
    """Build the report header."""
    lines = []
    lines.append("=" * 90)
    lines.append("            GMTSAR InSAR Processing Pipeline — Project Summary Report")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"  Generated:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Project root:     {project_root}")
    lines.append(f"  Orbit:            {orbit}")
    lines.append(f"  Satellite:        Sentinel-1")
    if config.get("relative_orbit"):
        lines.append(f"  Track:            {config['relative_orbit']}")
    if config.get("start_date") and config.get("end_date"):
        lines.append(f"  Date range:       {config['start_date']} — {config['end_date']}")
    if config.get("polygon"):
        lines.append(f"  AOI polygon:      {config['polygon'][:80]}...")
    if config.get("bbox"):
        bbox = config['bbox']
        if isinstance(bbox, list) and len(bbox) == 4:
            lines.append(f"  DEM bbox:         [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    lines.append("")
    return lines


def build_pipeline_status(project_root: Path) -> List[str]:
    """Build pipeline completion status section."""
    lines = []
    lines.append("-" * 90)
    lines.append("  PIPELINE STATUS")
    lines.append("-" * 90)
    lines.append("")

    state_path = project_root / "wrapper_meta" / "state" / "project_state.json"
    state = load_json(state_path)

    stage_names = {
        1: "Create project directory tree",
        2: "Download Sentinel-1 data",
        3: "Create DEM",
        4: "Download orbits & reframe",
        5: "Preprocess subswaths (align to master)",
        6: "Generate interferograms",
        7: "Merge subswaths",
        8: "Phase unwrapping (SNAPHU)",
        9: "SBAS time series inversion",
        10: "Post-SBAS (projection & visualization)",
    }

    completed = state.get("completed_stages", []) if state else []

    for num in range(1, 11):
        stage_key = f"stage_{num:02d}"
        status = "DONE" if stage_key in completed else "    "
        lines.append(f"    [{status}]  Stage {num:2d}: {stage_names[num]}")

    lines.append("")
    lines.append(f"  Completed: {len(completed)}/10 stages")
    lines.append("")
    return lines


def build_data_section(project_root: Path, orbit: str, config: Dict) -> List[str]:
    """Build data inventory section (Stage 2/4)."""
    lines = []
    lines.append("-" * 90)
    lines.append("  DATA INVENTORY")
    lines.append("-" * 90)
    lines.append("")

    data_dir = project_root / orbit / "data"

    # Count raw SAFE files
    safe_filelist = data_dir / "SAFE_filelist"
    total_safe = 0
    if safe_filelist.exists():
        total_safe = sum(1 for l in safe_filelist.read_text().strip().splitlines()
                         if l.strip() and l.strip().endswith(".SAFE"))
    lines.append(f"  SAFE files in filelist:       {total_safe}")

    # Count actual SAFE dirs on disk
    safe_on_disk = count_files(data_dir, "*.SAFE") if data_dir.exists() else 0
    lines.append(f"  SAFE directories on disk:     {safe_on_disk}")

    # Reframed data
    reframed_dir = None
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir() and d.name.startswith("F") and "_F" in d.name:
                reframed_dir = d
                break

    if reframed_dir:
        reframed_count = count_files(reframed_dir, "*.SAFE")
        lines.append(f"  Reframed SAFE ({reframed_dir.name}):  {reframed_count}")
    else:
        lines.append(f"  Reframed SAFE:                not found")

    lines.append("")
    return lines


def build_subswath_section(project_root: Path, orbit: str, config: Dict) -> List[str]:
    """Build per-subswath detailed section (Stages 5-6)."""
    lines = []
    lines.append("-" * 90)
    lines.append("  SUBSWATH DETAILS (Stages 5-6)")
    lines.append("-" * 90)
    lines.append("")

    subswaths = config.get("subswath_list", ["F1", "F2", "F3"])
    if isinstance(subswaths, str):
        subswaths = [subswaths]

    master_date = config.get("master_date", "unknown")
    lines.append(f"  Master date:  {master_date}")
    lines.append("")

    total_intfs = 0
    total_complete = 0

    for sw in subswaths:
        sw_dir = project_root / orbit / sw
        raw_dir = sw_dir / "raw"
        intf_all = sw_dir / "intf_all"

        lines.append(f"  --- {sw} ---")

        # Scenes from baseline_table
        bt_path = raw_dir / "baseline_table.dat"
        scenes = parse_baseline_table(bt_path)
        num_scenes = len(scenes)
        lines.append(f"    Scenes (baseline_table):    {num_scenes}")

        if scenes:
            dates = [s["decimal_year"] for s in scenes]
            min_year = min(dates)
            max_year = max(dates)
            lines.append(f"    Date range:                 {min_year:.1f} — {max_year:.1f}")

            bperps = [s["perpendicular_bl"] for s in scenes]
            lines.append(f"    Bperp range:                {min(bperps):.0f} to {max(bperps):.0f} m")

        # Interferograms from intf.in
        intf_in_path = sw_dir / "intf.in"
        pairs = parse_intf_in(intf_in_path)
        num_pairs = len(pairs)
        lines.append(f"    Interferogram pairs (intf.in): {num_pairs}")
        total_intfs += num_pairs

        # Completed interferograms (with phasefilt.grd AND corr.grd)
        complete = 0
        if intf_all.exists():
            for d in intf_all.iterdir():
                if d.is_dir() and "_" in d.name:
                    if (d / "phasefilt.grd").exists() and (d / "corr.grd").exists():
                        complete += 1
        lines.append(f"    Completed interferograms:   {complete}")
        total_complete += complete

        if num_pairs > 0:
            pct = 100 * complete / num_pairs
            lines.append(f"    Completion:                 {pct:.0f}%")

        # Temporal baselines from pairs
        if pairs and scenes:
            scene_map = {}
            for s in scenes:
                date_str = scene_stem_to_date(s["stem"])
                if date_str:
                    scene_map[s["stem"]] = s

            temp_baselines = []
            for ref, sec in pairs:
                if ref in scene_map and sec in scene_map:
                    dt = abs(scene_map[sec]["day"] - scene_map[ref]["day"])
                    temp_baselines.append(dt)
            if temp_baselines:
                lines.append(f"    Temporal baselines:         {min(temp_baselines)}-{max(temp_baselines)} days "
                             f"(mean {sum(temp_baselines)/len(temp_baselines):.0f})")

        lines.append("")

    lines.append(f"  TOTAL across subswaths:")
    lines.append(f"    Total intf pairs:           {total_intfs}")
    lines.append(f"    Total completed:            {total_complete}")
    lines.append("")
    return lines


def build_merge_section(project_root: Path, orbit: str, config: Dict) -> List[str]:
    """Build merge section (Stage 7)."""
    lines = []
    lines.append("-" * 90)
    lines.append("  MERGE (Stage 7)")
    lines.append("-" * 90)
    lines.append("")

    merge_dir = project_root / orbit / "merge"
    merge_mode = config.get("merge_mode", 0)
    mode_labels = {0: "F1+F2+F3", 1: "F1+F2", 2: "F2+F3"}
    lines.append(f"  Merge mode:       {merge_mode} ({mode_labels.get(merge_mode, '?')})")

    if not merge_dir.exists():
        lines.append(f"  Merge directory:  NOT FOUND")
        lines.append("")
        return lines

    # intflist
    intflist = merge_dir / "intflist"
    intflist_count = 0
    if intflist.exists():
        intflist_count = sum(1 for l in intflist.read_text().strip().splitlines() if l.strip())
    lines.append(f"  intflist entries:  {intflist_count}")

    # Count merged directories
    total_merge_dirs = count_dirs(merge_dir)
    lines.append(f"  Merge directories on disk:  {total_merge_dirs}")

    # With phasefilt.grd
    with_phasefilt = count_dirs_with_file(merge_dir, "phasefilt.grd")
    lines.append(f"  With phasefilt.grd:         {with_phasefilt}")

    # With corr.grd
    with_corr = count_dirs_with_file(merge_dir, "corr.grd")
    lines.append(f"  With corr.grd:              {with_corr}")

    # Validation
    if intflist_count > 0 and with_phasefilt != intflist_count:
        lines.append(f"  ** MISMATCH: intflist has {intflist_count} but {with_phasefilt} have phasefilt.grd")
    elif intflist_count > 0:
        lines.append(f"  Merge completion:           100%")

    lines.append("")
    return lines


def build_unwrap_section(project_root: Path, orbit: str, config: Dict) -> List[str]:
    """Build unwrap section (Stage 8)."""
    lines = []
    lines.append("-" * 90)
    lines.append("  PHASE UNWRAPPING (Stage 8)")
    lines.append("-" * 90)
    lines.append("")

    merge_dir = project_root / orbit / "merge"
    lines.append(f"  Coherence threshold (mask_def): {config.get('coherence_threshold', '?')}")
    lines.append(f"  Correlation threshold (snaphu):  {config.get('corr_threshold', '?')}")
    lines.append(f"  Max discontinuity threshold:     {config.get('max_dis_threshold', '?')}")
    lines.append(f"  Use landmask:                    {config.get('use_landmask', False)}")
    lines.append(f"  Use mask_def:                    {config.get('use_mask_def', True)}")

    # corr_stack.grd, mask_def.grd
    corr_stack = merge_dir / "corr_stack.grd"
    mask_def = merge_dir / "mask_def.grd"
    landmask = merge_dir / "landmask_ra.grd"
    lines.append(f"  corr_stack.grd:   {'EXISTS' if corr_stack.exists() else 'MISSING'}")
    lines.append(f"  mask_def.grd:     {'EXISTS' if mask_def.exists() else 'MISSING'}")
    lines.append(f"  landmask_ra.grd:  {'EXISTS' if landmask.exists() else 'MISSING'}")

    # Count unwrapped
    with_unwrap = count_dirs_with_file(merge_dir, "unwrap.grd")
    total_merge_dirs = count_dirs_with_file(merge_dir, "phasefilt.grd")
    lines.append(f"  Interferograms with unwrap.grd:  {with_unwrap}")
    lines.append(f"  Total merged interferograms:     {total_merge_dirs}")

    if total_merge_dirs > 0:
        pct = 100 * with_unwrap / total_merge_dirs
        lines.append(f"  Unwrap completion:               {pct:.0f}%")
        failed = total_merge_dirs - with_unwrap
        if failed > 0:
            lines.append(f"  Failed/missing unwrap:           {failed}")

    lines.append("")
    return lines


def build_sbas_section(project_root: Path, orbit: str) -> List[str]:
    """Build SBAS section (Stage 9)."""
    lines = []
    lines.append("-" * 90)
    lines.append("  SBAS TIME SERIES (Stage 9)")
    lines.append("-" * 90)
    lines.append("")

    sbas_dir = project_root / orbit / "SBAS"
    if not sbas_dir.exists():
        lines.append(f"  SBAS directory:   NOT FOUND")
        lines.append("")
        return lines

    # intf.tab
    intf_tab = sbas_dir / "intf.tab"
    num_intfs = 0
    if intf_tab.exists():
        num_intfs = sum(1 for l in intf_tab.read_text().strip().splitlines() if l.strip())
    lines.append(f"  Interferograms in intf.tab:  {num_intfs}")

    # scene.tab
    scene_tab = sbas_dir / "scene.tab"
    num_scenes = 0
    if scene_tab.exists():
        num_scenes = sum(1 for l in scene_tab.read_text().strip().splitlines() if l.strip())
    lines.append(f"  Scenes in scene.tab:        {num_scenes}")

    # Output grids
    vel_grd = sbas_dir / "vel.grd"
    rms_grd = sbas_dir / "rms.grd"
    dem_err = sbas_dir / "dem_err.grd"
    lines.append(f"  vel.grd:     {'EXISTS' if vel_grd.exists() else 'MISSING'}")
    lines.append(f"  rms.grd:     {'EXISTS' if rms_grd.exists() else 'MISSING'}")
    lines.append(f"  dem_err.grd: {'EXISTS' if dem_err.exists() else 'MISSING'}")

    # Displacement grids
    disp_grids = list(sbas_dir.glob("disp_*.grd"))
    disp_ra = [g for g in disp_grids if not g.name.endswith("_ll.grd")]
    disp_ll = [g for g in disp_grids if g.name.endswith("_ll.grd")]
    lines.append(f"  Displacement grids (radar):  {len(disp_ra)}")
    lines.append(f"  Displacement grids (ll):     {len(disp_ll)}")

    # Check SBAS log for parameters
    log_path = project_root / "wrapper_meta" / "logs" / f"step9_{orbit}_sbas.json"
    log = load_json(log_path)
    if log:
        sbas_inv = log.get("sbas_inversion", {})
        params = sbas_inv.get("parameters", {})
        if params:
            lines.append(f"  SBAS parameters:")
            lines.append(f"    Grid dimensions:  {params.get('xdim', '?')} x {params.get('ydim', '?')}")
            lines.append(f"    Wavelength:       {params.get('wavelength', '?')} m")
            lines.append(f"    Smooth factor:    {params.get('smooth', '?')}")
            lines.append(f"    Incidence:        {params.get('incidence', '?')}°")
        rc = sbas_inv.get("return_code")
        if rc is not None:
            lines.append(f"  SBAS return code:   {rc} ({'OK' if rc == 0 else 'FAILED'})")

    lines.append("")
    return lines


def build_post_sbas_section(project_root: Path, orbit: str) -> List[str]:
    """Build Post-SBAS section (Stage 10)."""
    lines = []
    lines.append("-" * 90)
    lines.append("  POST-SBAS OUTPUTS (Stage 10)")
    lines.append("-" * 90)
    lines.append("")

    sbas_dir = project_root / orbit / "SBAS"
    if not sbas_dir.exists():
        lines.append(f"  SBAS directory not found")
        lines.append("")
        return lines

    # Projected grids
    projected = {
        "vel_ll.grd": "Velocity (lat/lon)",
        "rms_ll.grd": "RMS residuals (lat/lon)",
        "dem_err_ll.grd": "DEM error (lat/lon)",
    }
    for fname, desc in projected.items():
        exists = (sbas_dir / fname).exists()
        lines.append(f"  {desc + ':':<35s} {'EXISTS' if exists else 'MISSING'}")

    # KML files
    kml_files = list(sbas_dir.glob("*.kml"))
    lines.append(f"  KML overlays:                     {len(kml_files)}")

    # Visualizations
    viz_dir = sbas_dir / "visualizations"
    if viz_dir.exists():
        png_count = count_files(viz_dir, "*.png")
        lines.append(f"  Visualization PNGs:               {png_count}")

    # Plots dir
    plots_dir = sbas_dir / "plots"
    if plots_dir.exists():
        point_plots = count_files(plots_dir, "point_*.png")
        lines.append(f"  GNSS point time series plots:     {point_plots}")

    lines.append("")
    return lines


def build_cross_validation(project_root: Path, orbit: str, config: Dict) -> List[str]:
    """Cross-validate counts between logs and actual files on disk."""
    lines = []
    lines.append("-" * 90)
    lines.append("  CROSS-VALIDATION")
    lines.append("-" * 90)
    lines.append("")

    issues = []
    ok_count = 0

    merge_dir = project_root / orbit / "merge"
    subswaths = config.get("subswath_list", ["F1", "F2", "F3"])
    if isinstance(subswaths, str):
        subswaths = [subswaths]

    # 1. Check intflist vs actual merged dirs with phasefilt
    intflist = merge_dir / "intflist"
    if intflist.exists():
        intflist_entries = set(l.strip() for l in intflist.read_text().strip().splitlines() if l.strip())
        actual_merged = set()
        if merge_dir.exists():
            for d in merge_dir.iterdir():
                if d.is_dir() and d.name.startswith("20") and (d / "phasefilt.grd").exists():
                    actual_merged.add(d.name)

        in_list_not_on_disk = intflist_entries - actual_merged
        on_disk_not_in_list = actual_merged - intflist_entries

        if in_list_not_on_disk:
            issues.append(f"  In intflist but missing phasefilt.grd: {len(in_list_not_on_disk)}")
        if on_disk_not_in_list:
            issues.append(f"  On disk with phasefilt.grd but not in intflist: {len(on_disk_not_in_list)}")
        if not in_list_not_on_disk and not on_disk_not_in_list:
            lines.append(f"  [OK] intflist matches merged dirs on disk ({len(actual_merged)} entries)")
            ok_count += 1

    # 2. Check unwrapped vs merged
    if merge_dir.exists():
        merged_count = count_dirs_with_file(merge_dir, "phasefilt.grd")
        unwrap_count = count_dirs_with_file(merge_dir, "unwrap.grd")
        if unwrap_count < merged_count:
            issues.append(f"  Unwrapped ({unwrap_count}) < Merged ({merged_count}): "
                          f"{merged_count - unwrap_count} interferograms missing unwrap.grd")
        else:
            lines.append(f"  [OK] All {merged_count} merged interferograms are unwrapped")
            ok_count += 1

    # 3. Check SBAS intf.tab vs unwrap count
    sbas_dir = project_root / orbit / "SBAS"
    intf_tab = sbas_dir / "intf.tab"
    if intf_tab.exists():
        sbas_intfs = sum(1 for l in intf_tab.read_text().strip().splitlines() if l.strip())
        if merge_dir.exists():
            unwrapped = count_dirs_with_file(merge_dir, "unwrap.grd")
            if sbas_intfs != unwrapped:
                issues.append(f"  SBAS intf.tab ({sbas_intfs}) != unwrapped on disk ({unwrapped})")
            else:
                lines.append(f"  [OK] SBAS intf.tab matches unwrapped count ({sbas_intfs})")
                ok_count += 1

    # 4. Check per-subswath intf counts are consistent
    intf_counts = {}
    for sw in subswaths:
        intf_in = project_root / orbit / sw / "intf.in"
        if intf_in.exists():
            intf_counts[sw] = sum(1 for l in intf_in.read_text().strip().splitlines()
                                  if l.strip() and ':' in l)

    if len(set(intf_counts.values())) > 1:
        parts = ", ".join(f"{sw}={n}" for sw, n in intf_counts.items())
        issues.append(f"  Subswath intf.in counts differ: {parts}")
    elif intf_counts:
        lines.append(f"  [OK] All subswaths have same intf.in count ({list(intf_counts.values())[0]})")
        ok_count += 1

    # 5. Log file checks
    log_dir = project_root / "wrapper_meta" / "logs"
    expected_logs = [
        f"step6_{orbit}_F1.json",
        f"step6_{orbit}_F2.json",
        f"step6_{orbit}_F3.json",
        f"step7_{orbit}.json",
        f"step8_{orbit}.json",
        f"step9_{orbit}_sbas.json",
        f"step10_{orbit}_post_sbas.json",
    ]
    missing_logs = [l for l in expected_logs if not (log_dir / l).exists()]
    if missing_logs:
        issues.append(f"  Missing log files: {', '.join(missing_logs)}")
    else:
        lines.append(f"  [OK] All expected log files present ({len(expected_logs)})")
        ok_count += 1

    # Print issues
    for issue in issues:
        lines.append(f"  [!!] {issue.strip()}")

    lines.append("")
    lines.append(f"  Checks passed: {ok_count}, Issues: {len(issues)}")
    lines.append("")
    return lines


def build_disk_usage(project_root: Path, orbit: str) -> List[str]:
    """Build disk usage summary."""
    lines = []
    lines.append("-" * 90)
    lines.append("  DISK USAGE (top-level directories)")
    lines.append("-" * 90)
    lines.append("")

    orbit_dir = project_root / orbit
    if not orbit_dir.exists():
        lines.append(f"  Orbit directory not found: {orbit_dir}")
        lines.append("")
        return lines

    dirs_to_check = ["data", "topo", "F1", "F2", "F3", "merge", "SBAS"]
    for dname in dirs_to_check:
        dpath = orbit_dir / dname
        if dpath.exists():
            # Just count files and subdirs rather than calculating full size
            # (size calculation can be very slow on NFS)
            if dname in ["F1", "F2", "F3"]:
                intf_count = count_dirs(dpath / "intf_all") if (dpath / "intf_all").exists() else 0
                lines.append(f"  {dname}/intf_all:   {intf_count} interferogram directories")
            elif dname == "merge":
                merge_dirs = count_dirs(dpath)
                lines.append(f"  merge/:          {merge_dirs} interferogram directories")
            elif dname == "SBAS":
                disp_count = len(list(dpath.glob("disp_*.grd")))
                lines.append(f"  SBAS/:           {disp_count} displacement grids")
            else:
                file_count = sum(1 for _ in dpath.iterdir()) if dpath.is_dir() else 0
                lines.append(f"  {dname}/:          {file_count} items")

    # wrapper_meta
    meta_dir = project_root / "wrapper_meta"
    if meta_dir.exists():
        log_count = count_files(meta_dir / "logs", "*.json")
        lines.append(f"  wrapper_meta/logs: {log_count} log files")

    lines.append("")
    return lines


def build_processing_timeline(project_root: Path, orbit: str) -> List[str]:
    """Build processing timeline from log timestamps."""
    lines = []
    lines.append("-" * 90)
    lines.append("  PROCESSING TIMELINE (from log timestamps)")
    lines.append("-" * 90)
    lines.append("")

    log_dir = project_root / "wrapper_meta" / "logs"
    if not log_dir.exists():
        lines.append(f"  No logs found")
        lines.append("")
        return lines

    timeline = []
    for log_file in sorted(log_dir.glob("*.json")):
        data = load_json(log_file)
        if data and "timestamp" in data:
            ts = data["timestamp"]
            step = data.get("step", "?")
            stage_name = data.get("stage_name", log_file.stem)
            timeline.append((ts, step, stage_name, log_file.name))

    for ts, step, name, fname in sorted(timeline):
        # Truncate timestamp to seconds
        ts_short = ts[:19] if len(ts) > 19 else ts
        lines.append(f"  {ts_short}  Stage {step:>2}: {name}")

    lines.append("")
    return lines


# =============================================================================
# MAIN REPORT GENERATOR
# =============================================================================

def generate_summary(project_root: Path, orbit: str) -> str:
    """
    Generate complete project summary report.

    Args:
        project_root: Project root directory
        orbit: Orbit directory name (asc/des)

    Returns:
        Complete report as a string
    """
    # Load config
    config = {}
    for config_name in ["config.yaml", "config.yml"]:
        config_path = project_root / config_name
        if config_path.exists():
            config = load_yaml_config(config_path)
            break
    else:
        # Try pattern config_*.yaml
        for p in project_root.glob("config_*.yaml"):
            config = load_yaml_config(p)
            break

    # Also merge project_state parameters
    state_path = project_root / "wrapper_meta" / "state" / "project_state.json"
    state = load_json(state_path)
    if state:
        for k, v in state.get("parameters", {}).items():
            if k not in config:
                config[k] = v

    # Build report sections
    all_lines = []
    all_lines.extend(build_header(project_root, orbit, config))
    all_lines.extend(build_pipeline_status(project_root))
    all_lines.extend(build_data_section(project_root, orbit, config))
    all_lines.extend(build_subswath_section(project_root, orbit, config))
    all_lines.extend(build_merge_section(project_root, orbit, config))
    all_lines.extend(build_unwrap_section(project_root, orbit, config))
    all_lines.extend(build_sbas_section(project_root, orbit))
    all_lines.extend(build_post_sbas_section(project_root, orbit))
    all_lines.extend(build_cross_validation(project_root, orbit, config))
    all_lines.extend(build_disk_usage(project_root, orbit))
    all_lines.extend(build_processing_timeline(project_root, orbit))

    # Footer
    all_lines.append("=" * 90)
    all_lines.append("  End of report")
    all_lines.append("=" * 90)

    return "\n".join(all_lines)


def run_summary(project_root: Path, orbit: str) -> Tuple[Path, str]:
    """
    Generate and save the project summary report.

    Args:
        project_root: Project root directory
        orbit: Orbit directory name

    Returns:
        (output_path, result_message)
    """
    logger.info("=" * 70)
    logger.info("STAGE 11: Generating Project Summary Report")
    logger.info("=" * 70)

    report = generate_summary(project_root, orbit)

    # Save report
    output_dir = project_root / "wrapper_meta"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "project_summary.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"  Summary report saved to: {output_path}")

    # Also save a timestamped copy in logs/
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = log_dir / f"project_summary_{orbit}_{ts}.txt"
    with open(timestamped, "w", encoding="utf-8") as f:
        f.write(report)

    # Write JSON metadata log
    meta = {
        "step": 11,
        "stage_name": "Project Summary",
        "orbit": orbit,
        "timestamp": datetime.datetime.now().isoformat(),
        "output_file": str(output_path),
        "timestamped_copy": str(timestamped),
    }
    meta_path = log_dir / f"step11_{orbit}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    result_msg = f"Project summary saved to {output_path}\n"
    result_msg += f"Timestamped copy: {timestamped}\n"

    return output_path, result_msg


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GMTSAR Stage 11: Project Summary Report"
    )
    parser.add_argument("project_root", help="Path to project root directory")
    parser.add_argument("orbit", help="Orbit directory (asc/des)")
    parser.add_argument("--print", action="store_true", dest="print_report",
                        help="Print report to stdout in addition to saving")

    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    output_path, result_msg = run_summary(project_root, args.orbit)
    print(result_msg)

    if args.print_report:
        print()
        print(Path(output_path).read_text())


if __name__ == "__main__":
    main()
