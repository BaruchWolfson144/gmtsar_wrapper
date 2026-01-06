#!/usr/bin/env python3
"""
make_dir_tree_01.py

Create GMTSAR directory structure and optionally download Sentinel-1 data.

This script implements:
- Step 1: General Setup (directory structure)
- Step 3: Data Selection and Download (automated download from ASF)

Based on GMTSAR Sentinel-1 Time Series Tutorial.
"""
import os
from pathlib import Path
import json
import datetime
import argparse
import urllib.request
import urllib.parse
import csv
import subprocess
import sys

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
    project = create_project_structure(project_root, desc)
    return project


def download_asf_data(polygon: str, start_date: str, end_date: str,
                      relative_orbit: int, username: str, password: str,
                      output_dir: Path, polarization: str = "VV"):
    """
    Download Sentinel-1 data from ASF using their API.

    This implements Section 3b (Automated download) from the PDF.

    Args:
        polygon: WKT polygon string (e.g., "POLYGON((-157 18,-154.2 18,...))")
        start_date: Start date (e.g., "January+1,+2018")
        end_date: End date (e.g., "September+30,+2018")
        relative_orbit: Path/orbit number (e.g., 124)
        username: ASF EarthData username
        password: ASF EarthData password
        output_dir: Directory to save downloaded files
        polarization: Polarization mode (default: VV)

    Returns:
        dict: Download statistics and metadata
    """
    print(f"Searching ASF for Sentinel-1 data...")
    print(f"  Polygon: {polygon[:50]}...")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Relative orbit: {relative_orbit}")

    # Build ASF API query URL (from PDF page 8)
    base_url = "https://api.daac.asf.alaska.edu/services/search/param"
    params = {
        "platform": "Sentinel-1A,Sentinel-1B",
        "polygon": polygon,
        "processingLevel": "SLC",
        "relativeOrbit": str(relative_orbit),
        "start": start_date,
        "end": end_date,
        "output": "csv"
    }

    # Add polarization if not default
    if polarization != "VV":
        params["polarization"] = polarization

    query_url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # Download CSV listing
    csv_file = output_dir / "asf_search.csv"
    try:
        print(f"Querying ASF API...")
        urllib.request.urlretrieve(query_url, csv_file)
        print(f"  Saved search results to {csv_file}")
    except Exception as e:
        return {"error": f"Failed to query ASF API: {e}", "downloaded": 0}

    # Parse CSV to extract download URLs (column 27 in PDF example)
    download_urls = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header

        # Find the URL column (usually called 'URL' or similar)
        url_col_idx = None
        for idx, header in enumerate(headers):
            if 'URL' in header.upper() or 'DOWNLOAD' in header.upper():
                url_col_idx = idx
                break

        if url_col_idx is None:
            # Fallback to column 26 (0-indexed, column 27 in 1-indexed)
            url_col_idx = 26

        for row in reader:
            if len(row) > url_col_idx:
                url = row[url_col_idx].strip('"')
                if url and url.startswith('http'):
                    download_urls.append(url)

    print(f"Found {len(download_urls)} scenes to download")

    # Save URLs to data.list file
    data_list = output_dir / "data.list"
    with open(data_list, 'w') as f:
        for url in download_urls:
            f.write(f"{url}\n")
    print(f"  Saved URLs to {data_list}")

    # Download each file using wget
    downloaded_count = 0
    failed_count = 0

    for idx, url in enumerate(download_urls, 1):
        # Extract filename from URL
        filename = url.split('/')[-1]
        safe_name = filename.replace('.zip', '.SAFE') if '.zip' in filename else filename
        output_path = output_dir / safe_name

        # Skip if already exists
        if output_path.exists():
            print(f"  [{idx}/{len(download_urls)}] Skipping {safe_name} (already exists)")
            downloaded_count += 1
            continue

        print(f"  [{idx}/{len(download_urls)}] Downloading {safe_name}...")

        # Use wget with credentials
        wget_cmd = [
            'wget',
            '--http-user', username,
            '--http-passwd', password,
            '--no-check-certificate',
            '-O', str(output_path) + '.zip',
            url
        ]

        try:
            result = subprocess.run(wget_cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                # Unzip the file
                print(f"    Unzipping...")
                unzip_cmd = ['unzip', '-q', str(output_path) + '.zip', '-d', str(output_dir)]
                subprocess.run(unzip_cmd, check=True)

                # Remove zip file
                (output_path.parent / (output_path.name + '.zip')).unlink()

                downloaded_count += 1
                print(f"    ✓ Downloaded and extracted")
            else:
                print(f"    ✗ Download failed: {result.stderr}")
                failed_count += 1
        except subprocess.TimeoutExpired:
            print(f"    ✗ Download timeout")
            failed_count += 1
        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed_count += 1

    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query_params": params,
        "total_scenes": len(download_urls),
        "downloaded": downloaded_count,
        "failed": failed_count,
        "output_directory": str(output_dir),
        "csv_file": str(csv_file),
        "data_list": str(data_list)
    }

    print(f"\nDownload complete:")
    print(f"  Total scenes: {len(download_urls)}")
    print(f"  Downloaded: {downloaded_count}")
    print(f"  Failed: {failed_count}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="GMTSAR Step 1 & 3: Create directory structure and optionally download data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create directory structure only
  python make_dir_tree_01.py /path/to/project --orbit asc

  # Create structure and download data
  python make_dir_tree_01.py /path/to/project --orbit asc \\
      --download \\
      --polygon "POLYGON((-157 18,-154.2 18,-154.2 20.4,-157 20.4,-157 18))" \\
      --start "January+1,+2018" \\
      --end "September+30,+2018" \\
      --relative-orbit 124 \\
      --username YOUR_ASF_USERNAME \\
      --password YOUR_ASF_PASSWORD
        """
    )

    # Required arguments
    parser.add_argument("project_root", help="Path to project root directory")

    # Orbit selection
    parser.add_argument("--orbit", required=True, choices=["asc", "des"],
                       help="Orbit type: asc (ascending) or des (descending)")

    # Optional download arguments
    parser.add_argument("--download", action="store_true",
                       help="Download Sentinel-1 data from ASF")
    parser.add_argument("--polygon", help="WKT polygon for area of interest")
    parser.add_argument("--start", help="Start date (e.g., 'January+1,+2018')")
    parser.add_argument("--end", help="End date (e.g., 'September+30,+2018')")
    parser.add_argument("--relative-orbit", type=int, help="Relative orbit/path number")
    parser.add_argument("--username", help="ASF EarthData username")
    parser.add_argument("--password", help="ASF EarthData password")
    parser.add_argument("--polarization", default="VV", help="Polarization mode (default: VV)")

    args = parser.parse_args()

    # Resolve project root
    project_root = Path(args.project_root).expanduser().resolve()
    is_descending = (args.orbit == "des")

    # Step 1: Create directory structure
    print("=" * 60)
    print("GMTSAR Step 1: Creating directory structure")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Orbit type: {args.orbit}")
    print()

    project = run_create_project(project_root, is_descending)
    print(f"✓ Directory structure created at {project}")

    # Step 3: Download data (optional)
    if args.download:
        # Validate required download arguments
        required_download_args = ["polygon", "start", "end", "relative_orbit", "username", "password"]
        missing_args = [arg for arg in required_download_args if not getattr(args, arg.replace("-", "_"))]

        if missing_args:
            print("\n✗ Error: Missing required arguments for download:")
            for arg in missing_args:
                print(f"  --{arg}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("GMTSAR Step 3: Downloading Sentinel-1 data from ASF")
        print("=" * 60)

        # Determine output directory
        orbit_dir = project_root / args.orbit
        data_dir = orbit_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download data
        download_metadata = download_asf_data(
            polygon=args.polygon,
            start_date=args.start,
            end_date=args.end,
            relative_orbit=args.relative_orbit,
            username=args.username,
            password=args.password,
            output_dir=data_dir,
            polarization=args.polarization
        )

        # Save download metadata
        meta_dir = project_root / "wrapper_meta" / "logs"
        meta_dir.mkdir(parents=True, exist_ok=True)
        download_log = meta_dir / f"download_{args.orbit}.json"

        with open(download_log, 'w') as f:
            json.dump(download_metadata, indent=4, fp=f)

        print(f"\n✓ Download metadata saved to {download_log}")

        if "error" in download_metadata:
            print(f"\n✗ Download failed: {download_metadata['error']}")
            sys.exit(1)
    else:
        print("\n(Skipping data download - use --download flag to enable)")

    print("\n" + "=" * 60)
    print("✓ Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


