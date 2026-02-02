#!/usr/bin/env python3
"""
validate_pins.py - Validate pins.ll coordinates against actual SAFE footprints

This script prevents the reframing issue where pins extend beyond the coverage
of some frames, causing data loss. It scans all SAFE directories, extracts
their footprints from manifest.safe, and checks if pins.ll is within the
common coverage area.

Usage:
    python3 validate_pins.py /path/to/project_root asc
    python3 validate_pins.py /path/to/project_root asc --pins /path/to/pins.ll
    python3 validate_pins.py /path/to/project_root asc --suggest
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_manifest_coordinates(manifest_path: Path) -> dict:
    """
    Extract footprint coordinates from manifest.safe file.

    Returns dict with:
        - coords: list of (lat, lon) tuples
        - min_lat, max_lat, min_lon, max_lon
    """
    try:
        content = manifest_path.read_text()
        match = re.search(r'<gml:coordinates>([^<]+)</gml:coordinates>', content)
        if not match:
            return None

        coord_str = match.group(1)
        # Format: "lat,lon lat,lon lat,lon lat,lon"
        coords = []
        for pair in coord_str.strip().split():
            lat, lon = pair.split(',')
            coords.append((float(lat), float(lon)))

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]

        return {
            'coords': coords,
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }
    except Exception as e:
        print(f"Warning: Could not parse {manifest_path}: {e}", file=sys.stderr)
        return None


def extract_date_from_safe(safe_name: str) -> str:
    """Extract date (YYYYMMDD) from SAFE directory name."""
    match = re.search(r'_(\d{8})T', safe_name)
    return match.group(1) if match else None


def scan_safe_directories(data_dir: Path) -> dict:
    """
    Scan all SAFE directories and extract footprints.

    Returns dict keyed by date, with list of frame footprints for each date.
    """
    results = defaultdict(list)

    safe_dirs = sorted(data_dir.glob("S1*.SAFE"))
    print(f"Found {len(safe_dirs)} SAFE directories")

    for safe_dir in safe_dirs:
        manifest = safe_dir / "manifest.safe"
        if not manifest.exists():
            print(f"Warning: No manifest.safe in {safe_dir.name}", file=sys.stderr)
            continue

        footprint = parse_manifest_coordinates(manifest)
        if footprint:
            date = extract_date_from_safe(safe_dir.name)
            footprint['safe_name'] = safe_dir.name
            results[date].append(footprint)

    return results


def analyze_coverage(footprints_by_date: dict) -> dict:
    """
    Analyze coverage across all dates.

    For each date with multiple frames (stitched), we need the combined coverage.
    The overall valid region is the INTERSECTION of all dates' combined coverages.
    """
    date_coverages = {}

    for date, frames in footprints_by_date.items():
        if not frames:
            continue

        # Combined coverage for this date (union of frames)
        combined_min_lat = min(f['min_lat'] for f in frames)
        combined_max_lat = max(f['max_lat'] for f in frames)
        combined_min_lon = min(f['min_lon'] for f in frames)
        combined_max_lon = max(f['max_lon'] for f in frames)

        date_coverages[date] = {
            'min_lat': combined_min_lat,
            'max_lat': combined_max_lat,
            'min_lon': combined_min_lon,
            'max_lon': combined_max_lon,
            'num_frames': len(frames)
        }

    if not date_coverages:
        return None

    # Find intersection (common coverage across ALL dates)
    # The valid pins region is bounded by:
    # - max of all min_lats (southernmost point that ALL dates cover)
    # - min of all max_lats (northernmost point that ALL dates cover)

    safe_min_lat = max(dc['min_lat'] for dc in date_coverages.values())
    safe_max_lat = min(dc['max_lat'] for dc in date_coverages.values())
    safe_min_lon = max(dc['min_lon'] for dc in date_coverages.values())
    safe_max_lon = min(dc['max_lon'] for dc in date_coverages.values())

    # Find which dates are the limiting factors
    limiting_south = [d for d, dc in date_coverages.items() if dc['min_lat'] == safe_min_lat]
    limiting_north = [d for d, dc in date_coverages.items() if dc['max_lat'] == safe_max_lat]

    return {
        'safe_min_lat': safe_min_lat,
        'safe_max_lat': safe_max_lat,
        'safe_min_lon': safe_min_lon,
        'safe_max_lon': safe_max_lon,
        'num_dates': len(date_coverages),
        'date_coverages': date_coverages,
        'limiting_south_dates': limiting_south,
        'limiting_north_dates': limiting_north
    }


def read_pins_file(pins_path: Path) -> tuple:
    """
    Read pins.ll file (format: lon lat on each line).
    Returns (pin1_lat, pin2_lat, pin1_lon, pin2_lon) or None if error.
    """
    try:
        lines = pins_path.read_text().strip().split('\n')
        if len(lines) < 2:
            return None

        # Format: lon lat
        pin1_parts = lines[0].split()
        pin2_parts = lines[1].split()

        pin1_lon, pin1_lat = float(pin1_parts[0]), float(pin1_parts[1])
        pin2_lon, pin2_lat = float(pin2_parts[0]), float(pin2_parts[1])

        return {
            'pin1_lat': pin1_lat,
            'pin2_lat': pin2_lat,
            'pin1_lon': pin1_lon,
            'pin2_lon': pin2_lon,
            'south_pin': min(pin1_lat, pin2_lat),
            'north_pin': max(pin1_lat, pin2_lat)
        }
    except Exception as e:
        print(f"Error reading pins file: {e}", file=sys.stderr)
        return None


def validate_pins(pins: dict, coverage: dict, margin: float = 0.1) -> dict:
    """
    Validate pins against coverage.

    Args:
        pins: Dict with pin coordinates
        coverage: Dict with safe coverage bounds
        margin: Safety margin in degrees (default 0.1° ≈ 11km)

    Returns dict with validation results.
    """
    issues = []

    south_pin = pins['south_pin']
    north_pin = pins['north_pin']
    safe_south = coverage['safe_min_lat']
    safe_north = coverage['safe_max_lat']

    # Check southern pin
    if south_pin < safe_south:
        diff = safe_south - south_pin
        issues.append({
            'type': 'SOUTH_PIN_OUT_OF_BOUNDS',
            'severity': 'ERROR',
            'message': f"Southern pin ({south_pin:.4f}°) is {diff:.4f}° south of safe minimum ({safe_south:.4f}°)",
            'suggested_fix': f"Change southern pin to {safe_south + margin:.2f}° or higher"
        })
    elif south_pin < safe_south + margin:
        diff = south_pin - safe_south
        issues.append({
            'type': 'SOUTH_PIN_MARGINAL',
            'severity': 'WARNING',
            'message': f"Southern pin ({south_pin:.4f}°) is only {diff:.4f}° from safe minimum ({safe_south:.4f}°)",
            'suggested_fix': f"Consider changing southern pin to {safe_south + margin:.2f}° for safety margin"
        })

    # Check northern pin
    if north_pin > safe_north:
        diff = north_pin - safe_north
        affected_dates = coverage['limiting_north_dates']
        issues.append({
            'type': 'NORTH_PIN_OUT_OF_BOUNDS',
            'severity': 'ERROR',
            'message': f"Northern pin ({north_pin:.4f}°) is {diff:.4f}° north of safe maximum ({safe_north:.4f}°)",
            'affected_dates': affected_dates[:5],  # Show first 5
            'num_affected_dates': len([d for d, dc in coverage['date_coverages'].items()
                                       if dc['max_lat'] < north_pin]),
            'suggested_fix': f"Change northern pin to {safe_north - margin:.2f}° or lower"
        })
    elif north_pin > safe_north - margin:
        diff = safe_north - north_pin
        issues.append({
            'type': 'NORTH_PIN_MARGINAL',
            'severity': 'WARNING',
            'message': f"Northern pin ({north_pin:.4f}°) is only {diff:.4f}° from safe maximum ({safe_north:.4f}°)",
            'suggested_fix': f"Consider changing northern pin to {safe_north - margin:.2f}° for safety margin"
        })

    return {
        'valid': len([i for i in issues if i['severity'] == 'ERROR']) == 0,
        'issues': issues,
        'pins': pins,
        'coverage': {
            'safe_south': safe_south,
            'safe_north': safe_north,
            'range': safe_north - safe_south
        }
    }


def print_report(validation_result: dict, coverage: dict, verbose: bool = False):
    """Print validation report."""

    print("\n" + "="*70)
    print("PINS VALIDATION REPORT")
    print("="*70)

    print(f"\n📊 Data Summary:")
    print(f"   Total dates: {coverage['num_dates']}")
    print(f"   Safe latitude range: {coverage['safe_min_lat']:.4f}° to {coverage['safe_max_lat']:.4f}°")
    print(f"   Coverage span: {coverage['safe_max_lat'] - coverage['safe_min_lat']:.2f}°")

    pins = validation_result['pins']
    print(f"\n📍 Configured Pins:")
    print(f"   Southern pin: {pins['south_pin']:.4f}°")
    print(f"   Northern pin: {pins['north_pin']:.4f}°")
    print(f"   Pins span: {pins['north_pin'] - pins['south_pin']:.2f}°")

    if validation_result['valid']:
        print(f"\n✅ VALIDATION PASSED - All pins within safe coverage")
    else:
        print(f"\n❌ VALIDATION FAILED - Issues found:")

    for issue in validation_result['issues']:
        icon = "❌" if issue['severity'] == 'ERROR' else "⚠️"
        print(f"\n{icon} {issue['type']}")
        print(f"   {issue['message']}")
        if 'num_affected_dates' in issue:
            print(f"   Dates that would fail: {issue['num_affected_dates']}")
            if 'affected_dates' in issue:
                print(f"   Example dates: {', '.join(issue['affected_dates'][:5])}")
        print(f"   💡 Suggested fix: {issue['suggested_fix']}")

    if verbose:
        print(f"\n📋 Limiting Dates:")
        print(f"   Dates limiting SOUTH boundary: {', '.join(coverage['limiting_south_dates'][:5])}")
        print(f"   Dates limiting NORTH boundary: {', '.join(coverage['limiting_north_dates'][:5])}")

    print("\n" + "="*70)


def suggest_pins(coverage: dict, margin: float = 0.1):
    """Suggest optimal pin values."""
    suggested_south = coverage['safe_min_lat'] + margin
    suggested_north = coverage['safe_max_lat'] - margin

    print(f"\n💡 SUGGESTED PINS (with {margin}° safety margin):")
    print(f"   Southern pin: {suggested_south:.2f}°")
    print(f"   Northern pin: {suggested_north:.2f}°")
    print(f"   Coverage: {suggested_north - suggested_south:.2f}°")
    print(f"\n   For pins.ll file:")
    print(f"   32.5 {suggested_south:.1f}")
    print(f"   32.5 {suggested_north:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate pins.ll coordinates against SAFE footprints'
    )
    parser.add_argument('project_root', type=str, help='Path to project root')
    parser.add_argument('orbit', type=str, default='asc', help='Orbit (asc/des)')
    parser.add_argument('--pins', type=str, help='Path to pins.ll (default: {project_root}/{orbit}/reframed/pins.ll)')
    parser.add_argument('--suggest', action='store_true', help='Suggest optimal pin values')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--margin', type=float, default=0.1, help='Safety margin in degrees (default: 0.1)')
    parser.add_argument('--exit-on-error', action='store_true', help='Exit with code 1 if validation fails')

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    orbit = args.orbit

    data_dir = project_root / orbit / "data"
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Scan SAFE directories
    print(f"Scanning {data_dir}...")
    footprints = scan_safe_directories(data_dir)

    if not footprints:
        print("Error: No SAFE directories found with valid footprints", file=sys.stderr)
        sys.exit(1)

    # Analyze coverage
    coverage = analyze_coverage(footprints)

    if coverage is None:
        print("Error: Could not analyze coverage", file=sys.stderr)
        sys.exit(1)

    # Read pins file
    if args.pins:
        pins_path = Path(args.pins)
    else:
        pins_path = project_root / orbit / "reframed" / "pins.ll"

    if args.suggest:
        suggest_pins(coverage, args.margin)
        if not pins_path.exists():
            print(f"\nNote: pins.ll not found at {pins_path}")
            sys.exit(0)

    if not pins_path.exists():
        print(f"Error: Pins file not found: {pins_path}", file=sys.stderr)
        print(f"Use --suggest to get recommended values")
        sys.exit(1)

    print(f"Reading pins from: {pins_path}")
    pins = read_pins_file(pins_path)

    if pins is None:
        print("Error: Could not parse pins file", file=sys.stderr)
        sys.exit(1)

    # Validate
    result = validate_pins(pins, coverage, args.margin)
    print_report(result, coverage, args.verbose)

    if args.suggest:
        suggest_pins(coverage, args.margin)

    if args.exit_on_error and not result['valid']:
        sys.exit(1)


if __name__ == "__main__":
    main()
