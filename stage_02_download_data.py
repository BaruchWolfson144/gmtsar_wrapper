#!/usr/bin/env python3
"""
stage_02_download_data.py

Download Sentinel-1 data from ASF (Alaska Satellite Facility) archive.

This script implements Stage 2 from the GMTSAR Sentinel-1 Time Series Tutorial:
- Automated download of Sentinel-1 SLC data using ASF API
- Query filtering by polygon, date range, orbit, and polarization
- Batch download with authentication
- Progress tracking and metadata logging

The download process follows Section 2b (Automated Download) from sentinel_time_series.pdf.

References:
    - ASF API: https://api.daac.asf.alaska.edu/services/search/param
    - GMTSAR Tutorial: sentinel_time_series.pdf (Section 2)
"""

import json
import logging
import datetime
import urllib.request
import urllib.parse
import csv
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


def download_asf_data(polygon: str, start_date: str, end_date: str,
                      relative_orbit: int, username: str, password: str,
                      output_dir: Path, polarization: str = "VV") -> Dict[str, Any]:
    """
    Download Sentinel-1 data from ASF using their API.

    This implements Section 2b (Automated download) from sentinel_time_series.pdf.

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
        dict: Download statistics and metadata including:
            - timestamp: ISO timestamp of download
            - query_params: ASF API query parameters used
            - total_scenes: Number of scenes found
            - downloaded: Number of scenes successfully downloaded
            - failed: Number of failed downloads
            - output_directory: Path to download directory
            - csv_file: Path to ASF search results CSV
            - data_list: Path to data.list file with URLs

    Raises:
        Exception: If ASF API query fails or download errors occur

    Example:
        >>> metadata = download_asf_data(
        ...     polygon="POLYGON((-157 18,-154.2 18,-154.2 20.4,-157 20.4,-157 18))",
        ...     start_date="January+1,+2023",
        ...     end_date="March+31,+2023",
        ...     relative_orbit=124,
        ...     username="myuser",
        ...     password="mypass",
        ...     output_dir=Path("/data/sentinel1"),
        ...     polarization="VV"
        ... )
        >>> print(f"Downloaded {metadata['downloaded']} scenes")
    """
    logger.info(f"Searching ASF for Sentinel-1 data...")
    logger.info(f"  Polygon: {polygon[:50]}...")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Relative orbit: {relative_orbit}")

    # Build ASF API query URL (from sentinel_time_series.pdf)
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
        logger.info(f"Querying ASF API...")
        urllib.request.urlretrieve(query_url, csv_file)
        logger.info(f"  Saved search results to {csv_file}")
    except Exception as e:
        error_msg = f"Failed to query ASF API: {e}"
        logger.error(error_msg)
        return {"error": error_msg, "downloaded": 0, "total_scenes": 0, "failed": 0}

    # Parse CSV to extract download URLs
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

    logger.info(f"Found {len(download_urls)} scenes to download")

    # Save URLs to data.list file
    data_list = output_dir / "data.list"
    with open(data_list, 'w') as f:
        for url in download_urls:
            f.write(f"{url}\n")
    logger.info(f"  Saved URLs to {data_list}")

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
            logger.info(f"  [{idx}/{len(download_urls)}] Skipping {safe_name} (already exists)")
            downloaded_count += 1
            continue

        logger.info(f"  [{idx}/{len(download_urls)}] Downloading {safe_name}...")

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
                logger.info(f"    Unzipping...")
                unzip_cmd = ['unzip', '-q', str(output_path) + '.zip', '-d', str(output_dir)]
                subprocess.run(unzip_cmd, check=True)

                # Remove zip file
                (output_path.parent / (output_path.name + '.zip')).unlink()

                downloaded_count += 1
                logger.info(f"    ✓ Downloaded and extracted")
            else:
                logger.warning(f"    ✗ Download failed: {result.stderr}")
                failed_count += 1
        except subprocess.TimeoutExpired:
            logger.warning(f"    ✗ Download timeout")
            failed_count += 1
        except Exception as e:
            logger.warning(f"    ✗ Error: {e}")
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

    logger.info(f"\nDownload complete:")
    logger.info(f"  Total scenes: {len(download_urls)}")
    logger.info(f"  Downloaded: {downloaded_count}")
    logger.info(f"  Failed: {failed_count}")

    return metadata


def run_download_data(
    project_root: Path,
    polygon: str,
    start_date: str,
    end_date: str,
    relative_orbit: int,
    username: str,
    password: str,
    orbit: str = "asc",
    polarization: str = "VV"
) -> tuple[Path, Dict[str, Any]]:
    """
    Download Sentinel-1 SLC data from ASF archive.

    This is the main entry point for Stage 02, which downloads raw SAR data
    from the Alaska Satellite Facility based on spatial and temporal criteria.

    Args:
        project_root: Root directory of the GMTSAR project
        polygon: WKT polygon string defining area of interest
                 Example: "POLYGON((lon1 lat1, lon2 lat2, ...))"
        start_date: Start date in ASF format (e.g., "January+1,+2023")
        end_date: End date in ASF format (e.g., "December+31,+2023")
        relative_orbit: Satellite relative orbit/path number (e.g., 87)
        username: ASF EarthData username for authentication
        password: ASF EarthData password for authentication
        orbit: Orbit type - "asc" (ascending) or "des" (descending). Default: "asc"
        polarization: SAR polarization mode. Default: "VV"
                     Options: "VV", "VH", "HH", "HV", "VV+VH", "HH+HV"

    Returns:
        tuple: (log_path, metadata_dict)
            - log_path: Path to the JSON log file containing download metadata
            - metadata_dict: Dictionary with download statistics and results
              {
                  "timestamp": "ISO timestamp",
                  "query_params": {...},
                  "total_scenes": int,
                  "downloaded": int,
                  "failed": int,
                  "output_directory": str,
                  "csv_file": str,
                  "data_list": str
              }

    Raises:
        ValueError: If required parameters are missing or invalid
        RuntimeError: If download fails or returns an error

    Example:
        >>> logp, metadata = run_download_data(
        ...     project_root=Path("/data/project"),
        ...     polygon="POLYGON((34.5 31.0, 35.5 31.0, 35.5 32.0, 34.5 32.0, 34.5 31.0))",
        ...     start_date="January+1,+2023",
        ...     end_date="March+31,+2023",
        ...     relative_orbit=87,
        ...     username="myuser",
        ...     password="mypass",
        ...     orbit="asc",
        ...     polarization="VV"
        ... )
        >>> print(f"Downloaded {metadata['downloaded']} scenes")

    Notes:
        - Requires valid ASF EarthData credentials (free registration at urs.earthdata.nasa.gov)
        - Downloaded files are stored in {project_root}/{orbit}/data/
        - Creates a CSV query results file and a data.list file with download URLs
        - Already downloaded files are automatically skipped
        - Each scene is downloaded as a .zip file and automatically extracted to .SAFE format
        - Download progress is logged in real-time
        - Failed downloads are tracked in the metadata

    GMTSAR Context:
        This stage follows Stage 01 (directory creation).
        The downloaded .SAFE directories will be used in Stage 03 (DEM preparation),
        Stage 04 (orbit download), and Stage 05 (preprocessing and master selection).
    """
    project_root = Path(project_root)

    # Validate orbit parameter
    if orbit not in ["asc", "des"]:
        raise ValueError(f"Invalid orbit '{orbit}'. Must be 'asc' or 'des'")

    # Determine output directory based on orbit
    orbit_dir = project_root / orbit
    data_dir = orbit_dir / "data"

    # Ensure data directory exists
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        logger.info("Creating data directory...")
        data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Sentinel-1 data to: {data_dir}")
    logger.info(f"Query parameters:")
    logger.info(f"  Area: {polygon[:60]}...")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Relative orbit: {relative_orbit}")
    logger.info(f"  Polarization: {polarization}")

    # Call the download function from make_dir_tree_01
    metadata = download_asf_data(
        polygon=polygon,
        start_date=start_date,
        end_date=end_date,
        relative_orbit=relative_orbit,
        username=username,
        password=password,
        output_dir=data_dir,
        polarization=polarization
    )

    # Check for errors in download
    if "error" in metadata:
        error_msg = metadata["error"]
        logger.error(f"Download failed: {error_msg}")
        raise RuntimeError(f"ASF data download failed: {error_msg}")

    # Save metadata log
    log_dir = project_root / "wrapper_meta" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"stage_02_download_data_{orbit}.json"

    with open(log_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Download metadata saved to: {log_path}")
    logger.info(f"Total scenes: {metadata['total_scenes']}")
    logger.info(f"Successfully downloaded: {metadata['downloaded']}")
    logger.info(f"Failed: {metadata['failed']}")

    return log_path, metadata


def validate_download_parameters(
    polygon: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    relative_orbit: Optional[int],
    username: Optional[str],
    password: Optional[str]
) -> list[str]:
    """
    Validate that all required download parameters are provided.

    Args:
        polygon: WKT polygon string
        start_date: Start date string
        end_date: End date string
        relative_orbit: Orbit number
        username: ASF username
        password: ASF password

    Returns:
        List of missing parameter names (empty if all valid)

    Example:
        >>> missing = validate_download_parameters(None, "Jan+1,+2023", None, 87, "user", "pass")
        >>> print(missing)
        ['polygon', 'end_date']
    """
    required_params = {
        "polygon": polygon,
        "start_date": start_date,
        "end_date": end_date,
        "relative_orbit": relative_orbit,
        "username": username,
        "password": password
    }

    missing = [name for name, value in required_params.items() if value is None]
    return missing


# Main entry point for standalone execution
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="GMTSAR Stage 02: Download Sentinel-1 data from ASF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python stage_02_download_data.py \\
      --project-root /path/to/project \\
      --polygon "POLYGON((34.5 31.0, 35.5 31.0, 35.5 32.0, 34.5 32.0, 34.5 31.0))" \\
      --start "January+1,+2023" \\
      --end "December+31,+2023" \\
      --relative-orbit 87 \\
      --username YOUR_USERNAME \\
      --password YOUR_PASSWORD \\
      --orbit asc \\
      --polarization VV
        """
    )

    parser.add_argument("--project-root", required=True,
                       help="Path to project root directory")
    parser.add_argument("--polygon", required=True,
                       help="WKT polygon for area of interest")
    parser.add_argument("--start", required=True,
                       help="Start date (e.g., 'January+1,+2023')")
    parser.add_argument("--end", required=True,
                       help="End date (e.g., 'December+31,+2023')")
    parser.add_argument("--relative-orbit", type=int, required=True,
                       help="Relative orbit/path number")
    parser.add_argument("--username", required=True,
                       help="ASF EarthData username")
    parser.add_argument("--password", required=True,
                       help="ASF EarthData password")
    parser.add_argument("--orbit", default="asc", choices=["asc", "des"],
                       help="Orbit type (default: asc)")
    parser.add_argument("--polarization", default="VV",
                       help="Polarization mode (default: VV)")

    args = parser.parse_args()

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("GMTSAR STAGE 02: Downloading Sentinel-1 Data from ASF")
    print("=" * 70)

    try:
        log_path, metadata = run_download_data(
            project_root=Path(args.project_root),
            polygon=args.polygon,
            start_date=args.start,
            end_date=args.end,
            relative_orbit=args.relative_orbit,
            username=args.username,
            password=args.password,
            orbit=args.orbit,
            polarization=args.polarization
        )

        print("\n" + "=" * 70)
        print("✓ DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"Log file: {log_path}")
        print(f"Downloaded: {metadata['downloaded']}/{metadata['total_scenes']} scenes")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Stage 02 failed: {e}", exc_info=True)
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)
