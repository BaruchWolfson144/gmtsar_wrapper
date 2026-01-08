#!/usr/bin/env python3
"""
make_dir_tree_01.py

Create GMTSAR directory structure (Stage 01).

This script implements Step 1: General Setup (directory structure creation).

For data download functionality, use stage_02_download_data.py or run the
full pipeline with main.py.

Based on GMTSAR Sentinel-1 Time Series Tutorial.
"""
from pathlib import Path
import json
import datetime
import argparse

GMTSAR_SUBDIRS = [
    "data",
    "orbit",
    "reframed",
    "topo",
    "F1/raw",
    "F1/topo",
    "F1/SLC",
    "F1/intf_in",
    "F2/raw",
    "F2/topo",
    "F2/SLC",
    "F2/intf_in",
    "F3/raw",
    "F3/topo",
    "F3/SLC",
    "F3/intf_in",
    "merge",
    "SBAS"
]

def create_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def create_gmtsar_tree(root: Path):
    for sub in GMTSAR_SUBDIRS:
        create_dir(root / sub)

def create_wrapper_meta(root: Path):
    meta = root / "wrapper_meta"
    create_dir(meta)
    create_dir(meta / "logs")
    create_dir(meta / "state")


    config = {
        "created_at": datetime.datetime.now().isoformat(),
        "version": "0.1",
        "description": "Metadata for GMTSAR Python wrapper"
    }
    with open(meta / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def create_project_structure(project_root: str, desc=False):
    root = Path(project_root)
    create_dir(root)

    #  ASCENDING
    if not desc:
        asc = root / "asc"
        create_dir(asc)
        create_gmtsar_tree(asc)

    #  DESENDING
    else:
        des = root / "des"
        create_dir(des)
        create_gmtsar_tree(des)

    # WRAPPERS META-DATA
    create_wrapper_meta(root)

    return root


def run_create_project(project_root: Path, desc: bool):
    """
    Create GMTSAR project directory structure.

    Args:
        project_root: Root directory for the project
        desc: True for descending orbit, False for ascending

    Returns:
        Path: The created project root directory
    """
    project = create_project_structure(project_root, desc)
    return project


def main():
    parser = argparse.ArgumentParser(
        description="GMTSAR Stage 01: Create directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create ascending orbit project structure
  python make_dir_tree_01.py /path/to/project --orbit asc

  # Create descending orbit project structure
  python make_dir_tree_01.py /path/to/project --orbit des

Note:
  For data download, use stage_02_download_data.py or run the full pipeline:
  python main.py --project-root /path/to/project --config config.yaml --sequential
        """
    )

    # Required arguments
    parser.add_argument("project_root", help="Path to project root directory")

    # Orbit selection
    parser.add_argument("--orbit", required=True, choices=["asc", "des"],
                       help="Orbit type: asc (ascending) or des (descending)")

    args = parser.parse_args()

    # Resolve project root
    project_root = Path(args.project_root).expanduser().resolve()
    is_descending = (args.orbit == "des")

    # Create directory structure
    print("=" * 60)
    print("GMTSAR Stage 01: Creating Directory Structure")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Orbit type: {args.orbit}")
    print()

    project = run_create_project(project_root, is_descending)

    print(f"✓ Directory structure created at {project}")
    print()
    print("Next steps:")
    print("  - Stage 02: Download data (stage_02_download_data.py)")
    print("  - Stage 03: Create DEM (make_dem_03.py)")
    print("  - Or run full pipeline: python main.py --config config.yaml --sequential")

    print("\n" + "=" * 60)
    print("✓ Stage 01 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


