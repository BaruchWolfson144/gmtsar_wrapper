#!/usr/bin/env python3
"""
main.py - GMTSAR InSAR Time Series Processing Pipeline

This is the main entry point for the GMTSAR Python wrapper that provides:
- Modular execution of individual processing stages
- Sequential automatic pipeline execution
- State management to track progress and pass data between stages
- Resume capability from last completed stage

The pipeline follows the workflow described in sentinel_time_series.pdf

Usage Examples:
    # Run full sequential pipeline with config file:
    python main.py --project-root /path/to/project --config config.yaml --sequential

    # Run specific stage:
    python main.py --project-root /path/to/project --stage 5

    # Resume from last completed stage:
    python main.py --project-root /path/to/project --resume

    # Run stages 5-7 only:
    python main.py --project-root /path/to/project --start-stage 5 --end-stage 7 --sequential
"""

import json
import logging
import argparse
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import processing stages
from make_dir_tree_01 import run_create_project
from stage_02_download_data import run_download_data, validate_download_parameters
from make_dem_03 import run_make_dem
from make_orbits_04 import run_download_orbits
from choose_master_05 import (
    run_preprocess_subswath,
    run_mode1_only,
    validate_master_across_subswaths,
    promote_master_and_run_mode2,
    get_master_date,
    is_mode1_complete,
    is_mode2_complete,
)
from run_interferograms_06 import run_intf, run_intf_single, copy_and_set_config, preparing_intf_list
from merge_intfs_07 import run_merge
from unwrap_intfs_08 import run_unwrap
from run_sbas_09 import run_sbas
from post_sbas_10 import run_post_sbas
from project_summary_11 import run_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectState:
    """
    Manages project state across processing stages.
    Stores results from completed stages to be used by subsequent stages.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.state_file = self.project_root / "wrapper_meta" / "state" / "project_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load existing state or create new state dictionary"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}. Creating new state.")

        return {
            "completed_stages": [],
            "parameters": {},
            "outputs": {}
        }

    def save_state(self):
        """Save current state to file"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        logger.debug(f"State saved to {self.state_file}")

    def mark_stage_complete(self, stage_name: str, outputs: Dict[str, Any]):
        """Mark a stage as completed and store its outputs"""
        if stage_name not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage_name)
        self.state["outputs"][stage_name] = outputs
        self.save_state()
        logger.info(f"Stage {stage_name} marked as complete")

    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage has been completed"""
        return stage_name in self.state["completed_stages"]

    def get_output(self, stage_name: str, key: str = None) -> Any:
        """Get output from a completed stage"""
        if key is None:
            return self.state["outputs"].get(stage_name, {})
        return self.state["outputs"].get(stage_name, {}).get(key)

    def set_parameter(self, key: str, value: Any):
        """Store a user-provided parameter"""
        self.state["parameters"][key] = value
        self.save_state()

    def get_parameter(self, key: str, default=None) -> Any:
        """Retrieve a stored parameter"""
        return self.state["parameters"].get(key, default)


class InSARPipeline:
    """
    Main InSAR processing pipeline with modular stage execution.

    Supports:
    - Sequential automatic execution
    - Individual stage execution
    - Resume from last completed stage
    - State management across stages
    """

    def __init__(self, project_root: str, config_file: Optional[str] = None):
        self.project_root = Path(project_root).expanduser().resolve()
        self.state = ProjectState(self.project_root)
        self.config = None

        # Load configuration if provided
        if config_file:
            self._load_config(config_file)

    def _load_config(self, config_file: str):
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_file).expanduser().resolve()

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            except ImportError:
                logger.error("PyYAML not installed. Please install it: pip install pyyaml")
                sys.exit(1)
        else:
            logger.error("Config file must be .json or .yaml")
            sys.exit(1)

        # Store all config parameters in state
        for key, value in self.config.items():
            if value is not None:
                self.state.set_parameter(key, value)

        logger.info(f"Configuration loaded from {config_path}")

    def _get_parallel_config(self) -> Dict[str, Any]:
        """Get parallelization configuration with defaults for backward compatibility."""
        parallel_config = self.state.get_parameter("parallel", {})

        # Default values ensure backward compatibility
        num_cores = parallel_config.get("num_cores", 6)
        if num_cores == "auto":
            num_cores = multiprocessing.cpu_count()

        return {
            "num_cores": min(int(num_cores), 16),  # Cap at 16 for safety
            "parallel_subswaths": parallel_config.get("parallel_subswaths", False),
        }

    def stage_01_create_project(self, orbit: Optional[str] = None) -> Dict[str, Any]:
        """Stage 01: Create project directory structure"""
        logger.info("=" * 70)
        logger.info("STAGE 01: Creating Project Directory Structure")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit")
        if not orbit:
            raise ValueError("Parameter 'orbit' (asc/des) is required for stage 01")

        is_descending = (orbit.lower() == "des")

        result = run_create_project(self.project_root, is_descending)

        outputs = {
            "orbit": orbit,
            "directories_created": True,
            "project_path": str(result)
        }

        self.state.mark_stage_complete("stage_01", outputs)
        logger.info(f"Project structure created at: {result}")
        return outputs

    def stage_03_make_dem(
        self,
        bbox: Optional[List[float]] = None,
        mode: Optional[int] = None,
        orbit: Optional[str] = None,
        make_dem_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stage 03: Create DEM grid"""
        logger.info("=" * 70)
        logger.info("STAGE 03: Creating DEM Grid")
        logger.info("=" * 70)

        # Get orbit parameter
        orbit = orbit or self.state.get_parameter("orbit", "asc")

        # Try to get bbox as a list first, or build it from individual parameters
        bbox = bbox or self.state.get_parameter("bbox")

        # If bbox not found as list, try to get from dem.* or individual parameters
        if not bbox:
            dem_config = self.state.get_parameter("dem")
            if dem_config and isinstance(dem_config, dict):
                # Extract from dem dict
                minlon = dem_config.get("minlon")
                maxlon = dem_config.get("maxlon")
                minlat = dem_config.get("minlat")
                maxlat = dem_config.get("maxlat")
                if all(v is not None for v in [minlon, maxlon, minlat, maxlat]):
                    bbox = [minlon, maxlon, minlat, maxlat]
                # Also get mode from dem dict if not provided
                if mode is None and "mode" in dem_config:
                    mode = dem_config["mode"]
            else:
                # Try individual parameters
                minlon = self.state.get_parameter("minlon")
                maxlon = self.state.get_parameter("maxlon")
                minlat = self.state.get_parameter("minlat")
                maxlat = self.state.get_parameter("maxlat")
                if all(v is not None for v in [minlon, maxlon, minlat, maxlat]):
                    bbox = [minlon, maxlon, minlat, maxlat]

        mode = mode or self.state.get_parameter("mode", 1)
        make_dem_path = make_dem_path or self.state.get_parameter("make_dem_path")

        if not bbox or len(bbox) != 4:
            raise ValueError("Parameter 'bbox' [minlon, maxlon, minlat, maxlat] is required")

        bbox_dict = {
            "minlon": bbox[0],
            "maxlon": bbox[1],
            "minlat": bbox[2],
            "maxlat": bbox[3]
        }

        logp, result_msg = run_make_dem(self.project_root, bbox_dict, mode, orbit=orbit, make_dem_path=make_dem_path)

        outputs = {
            "dem_created": True,
            "bbox": bbox,
            "mode": mode,
            "log_path": str(logp)
        }

        self.state.mark_stage_complete("stage_03", outputs)
        logger.info(f"DEM creation result:\n{result_msg}")
        return outputs

    def stage_02_download_data(
        self,
        polygon: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        relative_orbit: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        polarization: Optional[str] = None,
        orbit: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stage 02: Download Sentinel-1 data from ASF"""
        logger.info("=" * 70)
        logger.info("STAGE 02: Downloading Sentinel-1 Data from ASF")
        logger.info("=" * 70)

        # Get parameters from state/config
        polygon = polygon or self.state.get_parameter("polygon")
        start_date = start_date or self.state.get_parameter("start_date")
        end_date = end_date or self.state.get_parameter("end_date")
        relative_orbit = relative_orbit or self.state.get_parameter("relative_orbit")
        username = username or self.state.get_parameter("username") or self.state.get_parameter("asf_username")
        password = password or self.state.get_parameter("password") or self.state.get_parameter("asf_password")
        polarization = polarization or self.state.get_parameter("polarization", "VV")
        orbit = orbit or self.state.get_parameter("orbit", "asc")

        # Validate required parameters using the validation function
        missing = validate_download_parameters(
            polygon, start_date, end_date, relative_orbit, username, password
        )

        if missing:
            raise ValueError(
                f"Missing required parameters for Stage 02 (data download): {', '.join(missing)}. "
                f"Please provide them in config file or as function arguments."
            )

        # Call the download function
        logp, metadata = run_download_data(
            project_root=self.project_root,
            polygon=polygon,
            start_date=start_date,
            end_date=end_date,
            relative_orbit=relative_orbit,
            username=username,
            password=password,
            orbit=orbit,
            polarization=polarization
        )

        outputs = {
            "data_downloaded": True,
            "total_scenes": metadata["total_scenes"],
            "downloaded": metadata["downloaded"],
            "failed": metadata["failed"],
            "data_directory": metadata["output_directory"],
            "csv_file": metadata["csv_file"],
            "data_list": metadata["data_list"],
            "log_path": str(logp)
        }

        self.state.mark_stage_complete("stage_02", outputs)
        logger.info(f"Data download completed: {metadata['downloaded']}/{metadata['total_scenes']} scenes")
        return outputs

    def stage_04_download_orbits(
        self,
        orbit: Optional[str] = None,
        mode: Optional[int] = None,
        reframe: Optional[bool] = None,
        pin1_lon: Optional[float] = None,
        pin1_lat: Optional[float] = None,
        pin2_lon: Optional[float] = None,
        pin2_lat: Optional[float] = None
    ) -> Dict[str, Any]:
        """Stage 04: Download orbit files with optional reframing"""
        logger.info("=" * 70)
        logger.info("STAGE 04: Downloading Orbit Files")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit", "asc")
        mode = mode or self.state.get_parameter("orbit_mode", 1)
        reframe = reframe if reframe is not None else self.state.get_parameter("reframe", False)

        # Get parallel config for reframing
        parallel_config = None
        if reframe:
            parallel_settings = self.config.get("parallel", {})
            if parallel_settings.get("parallel_reframe", False):
                parallel_config = {
                    "parallel_reframe": True,
                    "reframe_workers": parallel_settings.get("reframe_workers", 4)
                }
                logger.info(f"Parallel reframe enabled: {parallel_config['reframe_workers']} workers")

        # Get pin coordinates if reframing
        if reframe:
            pin1_lon = pin1_lon or self.state.get_parameter("pin1_lon")
            pin1_lat = pin1_lat or self.state.get_parameter("pin1_lat")
            pin2_lon = pin2_lon or self.state.get_parameter("pin2_lon")
            pin2_lat = pin2_lat or self.state.get_parameter("pin2_lat")

            if not all([pin1_lon, pin1_lat, pin2_lon, pin2_lat]):
                raise ValueError("All pin coordinates required for reframing")

        success, result_msg, logp = run_download_orbits(
            self.project_root,
            orbit=orbit,
            mode=mode,
            reframe=reframe,
            pin1_lon=pin1_lon,
            pin1_lat=pin1_lat,
            pin2_lon=pin2_lon,
            pin2_lat=pin2_lat,
            parallel_config=parallel_config
        )

        if not success:
            raise RuntimeError(f"Orbit download failed: {result_msg}")

        outputs = {
            "orbits_downloaded": True,
            "mode": mode,
            "reframe": reframe,
            "log_path": str(logp) if logp else None
        }

        self.state.mark_stage_complete("stage_04", outputs)
        logger.info(f"Orbit download result: {result_msg}")
        return outputs

    def stage_05_preprocess_subswaths(
        self,
        orbit: Optional[str] = None,
        subswath_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Stage 05: Choose master and align images.

        Per GMTSAR documentation, the master image must be consistent across all subswaths.
        F1 determines the master date, which is then validated and used for F2 and F3.

        Supports resume: if a subswath already completed MODE 1 or MODE 2, it will be skipped.
        Master date can be set in config to skip automatic selection.

        Flow:
        1. Run MODE 1 for all subswaths (parallel if enabled) - skip completed
        2. Select master from config or F1
        3. Validate master date exists in all subswaths
        4. Run MODE 2 for all subswaths (parallel if enabled) - skip completed
        """
        logger.info("=" * 70)
        logger.info("STAGE 05: Preprocessing Subswaths and Choosing Master")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit", "asc")
        subswath_list = subswath_list or self.state.get_parameter("subswath_list", ["F1", "F2", "F3"])

        # Get parallel configuration
        parallel_config = self._get_parallel_config()
        parallel_subswaths = parallel_config.get("parallel_subswaths", False)

        # Check for master_date in config (allows skipping master selection)
        config_master_date = self.config.get("master_date") if self.config else None

        results = {}
        mode1_results = {}

        # ================================================================
        # PHASE 1: Run MODE 1 for all subswaths (baseline calculation)
        # Skip subswaths where MODE 1 already completed
        # ================================================================
        logger.info("-" * 50)
        logger.info("PHASE 1: Running MODE 1 (baseline calculation) for all subswaths")
        logger.info("-" * 50)

        # Separate completed and pending subswaths
        mode1_pending = []
        for sub in subswath_list:
            if is_mode1_complete(self.project_root, orbit, sub):
                logger.info(f"[{sub}] MODE 1 already complete, skipping")
                # Create placeholder result for skipped subswath
                mode1_results[sub] = {
                    "subswath": sub,
                    "master_info": {"master": f"S1_{config_master_date}_ALL_{sub}" if config_master_date else "unknown"},
                    "data_in_status": "ok (skipped)",
                    "mode1_status": "ok (skipped)",
                    "baseline_table_path": str(self.project_root / orbit / sub / "baseline_table.dat"),
                }
            else:
                mode1_pending.append(sub)

        # Run MODE 1 for pending subswaths
        if mode1_pending:
            if parallel_subswaths and len(mode1_pending) > 1:
                logger.info(f"Running MODE 1 in PARALLEL mode ({len(mode1_pending)} subswaths)")

                with ProcessPoolExecutor(max_workers=len(mode1_pending)) as executor:
                    futures = {
                        sub: executor.submit(run_mode1_only, self.project_root, orbit, sub)
                        for sub in mode1_pending
                    }

                    for sub, future in futures.items():
                        try:
                            mode1_results[sub] = future.result()
                            logger.info(f"[{sub}] MODE 1 completed successfully")
                        except Exception as e:
                            logger.error(f"[{sub}] MODE 1 failed: {e}")
                            raise RuntimeError(f"MODE 1 failed for {sub}: {e}")
            else:
                logger.info(f"Running MODE 1 SEQUENTIALLY ({len(mode1_pending)} subswaths)")
                for sub in mode1_pending:
                    logger.info(f"[{sub}] Running MODE 1...")
                    mode1_results[sub] = run_mode1_only(self.project_root, orbit, sub)
                    logger.info(f"[{sub}] MODE 1 completed successfully")
        else:
            logger.info("All subswaths already completed MODE 1")

        # ================================================================
        # PHASE 2: Select master date
        # Priority: config > state > calculate from F1
        # ================================================================
        logger.info("-" * 50)
        logger.info("PHASE 2: Selecting master date")
        logger.info("-" * 50)

        # Priority 1: From config file
        if config_master_date:
            master_date = str(config_master_date)
            logger.info(f"Using master_date from config: {master_date}")
        # Priority 2: From state (previous run)
        elif self.state.get_parameter("master_date"):
            master_date = self.state.get_parameter("master_date")
            logger.info(f"Using master_date from state: {master_date}")
        # Priority 3: Calculate from F1
        else:
            if "F1" not in mode1_results:
                raise ValueError("F1 must be in subswath_list to determine master image")

            f1_master_info = mode1_results["F1"]["master_info"]
            master_stem = f1_master_info["master"]  # e.g., "S1_20200104_ALL_F1"
            master_date = get_master_date(master_stem)  # e.g., "20200104"
            logger.info(f"F1 selected master: {master_stem} (date: {master_date})")

            # Log what other subswaths would have selected (for transparency)
            for sub in subswath_list:
                if sub != "F1" and sub in mode1_results:
                    sub_master_info = mode1_results[sub].get("master_info", {})
                    if "master" in sub_master_info:
                        sub_master = sub_master_info["master"]
                        sub_date = get_master_date(sub_master)
                        logger.info(f"[{sub}] Would have selected: {sub_master} (date: {sub_date})")

        # Save master_date to state for future reference
        self.state.set_parameter("master_date", master_date)

        # ================================================================
        # PHASE 3: Validate master across all subswaths
        # Skip if master_date came from config (user explicitly set it)
        # ================================================================
        logger.info("-" * 50)
        logger.info("PHASE 3: Validating master consistency across subswaths")
        logger.info("-" * 50)

        if config_master_date:
            logger.info("Skipping validation - master_date explicitly set in config")
        else:
            # Only validate if we have fresh mode1_results (not all skipped)
            if mode1_pending:
                validate_master_across_subswaths(
                    self.project_root,
                    orbit,
                    subswath_list,
                    master_date,
                    mode1_results
                )
                logger.info("Master validation PASSED - all subswaths have consistent data")
            else:
                logger.info("Skipping validation - all MODE 1 were skipped (using existing data)")

        # ================================================================
        # PHASE 4: Run MODE 2 for all subswaths (using master date)
        # Skip subswaths where MODE 2 already completed
        # ================================================================
        logger.info("-" * 50)
        logger.info(f"PHASE 4: Running MODE 2 (alignment) with master date {master_date}")
        logger.info("-" * 50)

        mode2_results = {}

        # Separate completed and pending subswaths
        mode2_pending = []
        for sub in subswath_list:
            if is_mode2_complete(self.project_root, orbit, sub):
                logger.info(f"[{sub}] MODE 2 already complete, skipping")
                mode2_results[sub] = {
                    "subswath": sub,
                    "master_promoted": "ok (skipped)",
                    "mode2_status": "ok (skipped)",
                }
            else:
                mode2_pending.append(sub)

        # Run MODE 2 for pending subswaths
        if mode2_pending:
            if parallel_subswaths and len(mode2_pending) > 1:
                logger.info(f"Running MODE 2 in PARALLEL mode ({len(mode2_pending)} subswaths)")

                with ProcessPoolExecutor(max_workers=len(mode2_pending)) as executor:
                    futures = {
                        sub: executor.submit(
                            promote_master_and_run_mode2,
                            self.project_root,
                            orbit,
                            sub,
                            master_date
                        )
                        for sub in mode2_pending
                    }

                    for sub, future in futures.items():
                        try:
                            mode2_results[sub] = future.result()
                            logger.info(f"[{sub}] MODE 2 completed successfully")
                        except Exception as e:
                            logger.error(f"[{sub}] MODE 2 failed: {e}")
                            raise RuntimeError(f"MODE 2 failed for {sub}: {e}")
            else:
                logger.info(f"Running MODE 2 SEQUENTIALLY ({len(mode2_pending)} subswaths)")
                for sub in mode2_pending:
                    logger.info(f"[{sub}] Running MODE 2...")
                    mode2_results[sub] = promote_master_and_run_mode2(
                        self.project_root, orbit, sub, master_date
                    )
                    logger.info(f"[{sub}] MODE 2 completed successfully")
        else:
            logger.info("All subswaths already completed MODE 2")

        # ================================================================
        # Compile results
        # ================================================================
        for sub in subswath_list:
            results[sub] = {
                "mode1": mode1_results.get(sub, {"status": "skipped"}),
                "mode2": mode2_results.get(sub, {"status": "skipped"}),
                "master_used": f"S1_{master_date}_ALL_{sub}",
                "status": "success"
            }

        # Create master_stem for compatibility
        master_stem = f"S1_{master_date}_ALL_F1"

        outputs = {
            "master": master_stem,
            "master_date": master_date,
            "subswaths_processed": subswath_list,
            "alignment_complete": True,
            "parallel_mode": parallel_subswaths,
            "results": results
        }

        # Store master for future stages
        self.state.set_parameter("master", master_stem)
        self.state.set_parameter("master_date", master_date)
        self.state.mark_stage_complete("stage_05", outputs)

        logger.info("=" * 70)
        logger.info(f"Stage 05 COMPLETED - Master date: {master_date}")
        logger.info("=" * 70)
        return outputs

    def stage_06_run_interferograms(
        self,
        orbit: Optional[str] = None,
        threshold_time: Optional[int] = None,
        threshold_baseline: Optional[float] = None,
        config_path: Optional[str] = None,
        master: Optional[str] = None,
        subswath_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Stage 06: Generate interferograms"""
        logger.info("=" * 70)
        logger.info("STAGE 06: Running Interferograms")
        logger.info("=" * 70)

        # Get parameters from state if not provided
        orbit = orbit or self.state.get_parameter("orbit", "asc")
        master = master or self.state.get_parameter("master")

        # Try to get parameters from interferograms dict first, then from root level
        interferograms_config = self.state.get_parameter("interferograms", {})
        threshold_time = threshold_time or interferograms_config.get("threshold_time") or self.state.get_parameter("threshold_time", 30)
        threshold_baseline = threshold_baseline or interferograms_config.get("threshold_baseline") or self.state.get_parameter("threshold_baseline", 100)
        config_path = config_path or interferograms_config.get("config_path") or self.state.get_parameter("config_path")

        # Try to get subswath_list from preprocessing dict first
        preprocessing_config = self.state.get_parameter("preprocessing", {})
        subswath_list = subswath_list or preprocessing_config.get("subswath_list") or self.state.get_parameter("subswath_list", ["F1", "F2", "F3"])

        if not master:
            raise ValueError("Master image not defined. Run stage 05 first or specify manually.")

        if not config_path:
            raise ValueError("config_path is required for stage 06. Provide batch_tops.config file path.")

        # Convert config_path to absolute Path if needed
        config_path = Path(config_path)
        if not config_path.is_absolute():
            # If relative, make it relative to project_root
            config_path = self.project_root / config_path
        config_path = config_path.resolve()

        # Get parallel configuration
        parallel_config = self._get_parallel_config()
        parallel_subswaths = parallel_config.get("parallel_subswaths", False)
        num_cores = parallel_config.get("num_cores", 6)

        results = {}

        if parallel_subswaths and len(subswath_list) > 1:
            logger.info(f"Running subswaths in PARALLEL mode ({len(subswath_list)} subswaths)")
            cores_per_subswath = max(2, num_cores // len(subswath_list))
            logger.info(f"Allocating {cores_per_subswath} cores per subswath")

            # STEP 1: Prepare intf.in for all subswaths (each uses its own baseline_table.dat)
            logger.info("Preparing intf.in files for all subswaths...")
            for sub in subswath_list:
                lines, years, baselines, intf_info = preparing_intf_list(
                    self.project_root, orbit, sub, threshold_time, threshold_baseline
                )
                logger.info(f"  [{sub}] Created intf.in with {intf_info['num_interferograms']} pairs")

            # STEP 2: Copy config to all subswaths BEFORE parallel execution
            logger.info("Copying batch_tops.config to all subswaths...")
            copy_and_set_config(self.project_root, orbit, Path(config_path), master, subswath_list)

            # STEP 3: Run interferograms in parallel using run_intf_single
            logger.info("Running interferogram generation in parallel...")
            with ProcessPoolExecutor(max_workers=len(subswath_list)) as executor:
                futures = {}
                for sub in subswath_list:
                    future = executor.submit(
                        run_intf_single,
                        self.project_root,
                        orbit,
                        sub,
                        threshold_time,
                        threshold_baseline,
                        Path(config_path),
                        master,
                        cores_per_subswath
                    )
                    futures[sub] = future

                # Collect results
                for sub, future in futures.items():
                    try:
                        logp = future.result()
                        results[sub] = {"log_path": str(logp) if logp else None, "status": "success"}
                        logger.info(f"Subswath {sub} completed successfully")
                    except Exception as e:
                        logger.error(f"Subswath {sub} failed: {e}")
                        results[sub] = {"log_path": None, "status": "failed", "error": str(e)}
        else:
            # Sequential processing (original behavior for backward compatibility)
            logger.info("Running subswaths SEQUENTIALLY")
            for sub in subswath_list:
                logger.info(f"Running interferograms for subswath {sub}")
                logp = run_intf(
                    self.project_root,
                    orbit,
                    sub,
                    threshold_time,
                    threshold_baseline,
                    Path(config_path),
                    master,
                    num_cores
                )
                results[sub] = {"log_path": str(logp) if logp else None}

        outputs = {
            "interferograms_complete": True,
            "subswaths": subswath_list,
            "master": master,
            "threshold_time": threshold_time,
            "threshold_baseline": threshold_baseline,
            "parallel_mode": parallel_subswaths,
            "num_cores": num_cores,
            "results": results
        }

        self.state.mark_stage_complete("stage_06", outputs)
        logger.info("Interferograms generation completed")
        return outputs

    def stage_07_merge(
        self,
        orbit: Optional[str] = None,
        master: Optional[str] = None,
        mode: Optional[int] = None
    ) -> Dict[str, Any]:
        """Stage 07: Merge interferograms across subswaths"""
        logger.info("=" * 70)
        logger.info("STAGE 07: Merging Interferograms")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit", "asc")
        master = master or self.state.get_parameter("master")
        mode = mode if mode is not None else self.state.get_parameter("merge_mode", 0)

        if not master:
            raise ValueError("Master image not defined. Run stage 05 first or specify manually.")

        logp = run_merge(self.project_root, orbit, master, mode)

        outputs = {
            "merge_complete": True,
            "mode": mode,
            "log_path": str(logp) if logp else None
        }

        self.state.mark_stage_complete("stage_07", outputs)
        logger.info("Interferogram merging completed")
        return outputs

    def stage_08_unwrap(
        self,
        orbit: Optional[str] = None,
        coherence_threshold: Optional[float] = None,
        corr_threshold: Optional[float] = None,
        max_dis_threshold: Optional[float] = None,
        use_landmask: Optional[bool] = None,
        use_mask_def: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Stage 08: Unwrap interferograms"""
        logger.info("=" * 70)
        logger.info("STAGE 08: Unwrapping Interferograms")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit", "asc")
        coherence_threshold = coherence_threshold or self.state.get_parameter("coherence_threshold", 0.075)
        corr_threshold = corr_threshold or self.state.get_parameter("corr_threshold", 0.01)
        max_dis_threshold = max_dis_threshold or self.state.get_parameter("max_dis_threshold", 40)
        use_landmask = use_landmask if use_landmask is not None else self.state.get_parameter("use_landmask", False)
        use_mask_def = use_mask_def if use_mask_def is not None else self.state.get_parameter("use_mask_def", True)

        # Get parallel configuration
        parallel_config = self._get_parallel_config()
        num_cores = parallel_config.get("num_cores", 6)
        logger.info(f"Using {num_cores} cores for unwrapping")

        logp, result_msg = run_unwrap(
            self.project_root,
            orbit,
            coherence_threshold=coherence_threshold,
            corr_threshold=corr_threshold,
            max_dis_threshold=max_dis_threshold,
            use_landmask=use_landmask,
            use_mask_def=use_mask_def,
            num_cores=num_cores
        )

        outputs = {
            "unwrap_complete": True,
            "coherence_threshold": coherence_threshold,
            "corr_threshold": corr_threshold,
            "max_dis_threshold": max_dis_threshold,
            "num_cores": num_cores,
            "log_path": str(logp) if logp else None
        }

        self.state.mark_stage_complete("stage_08", outputs)
        logger.info(f"Unwrapping completed:\n{result_msg}")
        return outputs

    def stage_09_run_sbas(
        self,
        orbit: Optional[str] = None,
        subswath: Optional[str] = None,
        smooth: Optional[float] = None,
        atm_iterations: Optional[int] = None,
        unwrap_file: Optional[str] = None,
        corr_file: Optional[str] = None,
        use_prep_script: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Stage 09: SBAS time series analysis"""
        logger.info("=" * 70)
        logger.info("STAGE 09: SBAS Time Series Analysis")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit", "asc")
        subswath = subswath or self.state.get_parameter("subswath", "F1")
        smooth = smooth if smooth is not None else self.state.get_parameter("smooth", 5.0)
        atm_iterations = atm_iterations if atm_iterations is not None else self.state.get_parameter("atm_iterations", 0)
        unwrap_file = unwrap_file or self.state.get_parameter("unwrap_file", "unwrap.grd")
        corr_file = corr_file or self.state.get_parameter("corr_file", "corr.grd")
        use_prep_script = use_prep_script if use_prep_script is not None else self.state.get_parameter("use_prep_script", True)

        logp, result_msg = run_sbas(
            self.project_root,
            orbit,
            subswath=subswath,
            unwrap_file=unwrap_file,
            corr_file=corr_file,
            smooth=smooth,
            atm_iterations=atm_iterations,
            use_prep_script=use_prep_script,
            use_mmap=True
        )

        outputs = {
            "sbas_complete": True,
            "subswath": subswath,
            "smooth": smooth,
            "atm_iterations": atm_iterations,
            "log_path": str(logp) if logp else None,
            "result_msg": result_msg
        }

        self.state.mark_stage_complete("stage_09", outputs)
        logger.info(f"SBAS completed:\n{result_msg}")
        return outputs

    def stage_10_post_sbas(
        self,
        orbit: Optional[str] = None,
        subswath: Optional[str] = None,
        vel_cmap: Optional[str] = None,
        vel_range: Optional[tuple] = None,
        project_disp: Optional[bool] = None,
        max_disp_grids: Optional[int] = None,
        gnss_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stage 10: Post-SBAS processing and visualization"""
        logger.info("=" * 70)
        logger.info("STAGE 10: Post-SBAS Processing and Visualization")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit", "asc")
        subswath = subswath or self.state.get_parameter("subswath", "F2")
        vel_cmap = vel_cmap or self.state.get_parameter("vel_cmap", "jet")
        vel_range = vel_range or self.state.get_parameter("vel_range", None)
        project_disp = project_disp if project_disp is not None else self.state.get_parameter("project_disp", True)
        max_disp_grids = max_disp_grids if max_disp_grids is not None else self.state.get_parameter("max_disp_grids", 50)
        gnss_file = gnss_file or self.state.get_parameter("gnss_file", None)

        logp, result_msg = run_post_sbas(
            self.project_root,
            orbit,
            subswath=subswath,
            vel_cmap=vel_cmap,
            vel_range=vel_range,
            project_disp=project_disp,
            max_disp_grids=max_disp_grids,
            gnss_file=gnss_file
        )

        outputs = {
            "post_sbas_complete": True,
            "subswath": subswath,
            "vel_cmap": vel_cmap,
            "project_disp": project_disp,
            "log_path": str(logp) if logp else None,
            "result_msg": result_msg
        }

        self.state.mark_stage_complete("stage_10", outputs)
        logger.info(f"Post-SBAS completed:\n{result_msg}")
        return outputs

    def stage_11_project_summary(
        self,
        orbit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage 11: Generate comprehensive project summary report"""
        logger.info("=" * 70)
        logger.info("STAGE 11: Project Summary Report")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit", "asc")

        output_path, result_msg = run_summary(self.project_root, orbit)

        outputs = {
            "summary_complete": True,
            "output_path": str(output_path),
            "result_msg": result_msg
        }

        self.state.mark_stage_complete("stage_11", outputs)
        logger.info(f"Summary report completed:\n{result_msg}")
        return outputs

    def run_sequential(self, start_stage: int = 1, end_stage: int = 11):
        """
        Run stages sequentially from start_stage to end_stage.
        Automatically passes outputs from previous stages as inputs to next stages.
        """
        logger.info("=" * 70)
        logger.info(f"RUNNING SEQUENTIAL PIPELINE: Stages {start_stage} - {end_stage}")
        logger.info(f"Project Root: {self.project_root}")
        logger.info("=" * 70)

        stage_methods = {
            1: self.stage_01_create_project,
            2: self.stage_02_download_data,
            3: self.stage_03_make_dem,
            4: self.stage_04_download_orbits,
            5: self.stage_05_preprocess_subswaths,
            6: self.stage_06_run_interferograms,
            7: self.stage_07_merge,
            8: self.stage_08_unwrap,
            9: self.stage_09_run_sbas,
            10: self.stage_10_post_sbas,
            11: self.stage_11_project_summary
        }

        for stage_num in range(start_stage, end_stage + 1):
            if stage_num in stage_methods:
                stage_name = f"stage_{stage_num:02d}"

                # Skip if already completed
                if self.state.is_stage_complete(stage_name):
                    logger.info(f"\nStage {stage_num:02d} already completed (skipping)")
                    continue

                logger.info(f"\n{'='*70}")
                logger.info(f"Executing Stage {stage_num:02d}")
                logger.info(f"{'='*70}\n")

                try:
                    stage_methods[stage_num]()
                except Exception as e:
                    logger.error(f"Error in stage {stage_num}: {e}")
                    logger.error("Pipeline halted. Fix the error and use --resume to continue.")
                    raise

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    def resume(self):
        """Resume pipeline from last completed stage"""
        completed = self.state.state["completed_stages"]

        if not completed:
            logger.info("No completed stages found. Starting from beginning.")
            self.run_sequential()
            return

        # Extract stage numbers
        stage_nums = [int(s.split('_')[1]) for s in completed]
        last_stage = max(stage_nums)

        logger.info(f"Last completed stage: {last_stage}")
        logger.info(f"Resuming from stage {last_stage + 1}")
        self.run_sequential(start_stage=last_stage + 1)

    def reset_from_stage(self, from_stage: int):
        """Reset completion status for stages >= from_stage, allowing re-run.

        This preserves stored parameters but clears completion flags and outputs
        for the specified stages. Useful when re-running with different thresholds
        (e.g., increasing temporal baseline from 180 to 360 days).
        """
        stages_to_reset = [f"stage_{s:02d}" for s in range(from_stage, 12)]
        reset_count = 0
        for stage_name in stages_to_reset:
            if stage_name in self.state.state["completed_stages"]:
                self.state.state["completed_stages"].remove(stage_name)
                reset_count += 1
            if stage_name in self.state.state["outputs"]:
                del self.state.state["outputs"][stage_name]
        self.state.save_state()
        logger.info(f"Reset {reset_count} stages (stage {from_stage} through 10)")
        logger.info(f"Completed stages remaining: {self.state.state['completed_stages']}")

    def show_status(self):
        """Display current pipeline status"""
        logger.info("=" * 70)
        logger.info("PIPELINE STATUS")
        logger.info("=" * 70)
        logger.info(f"Project Root: {self.project_root}")
        logger.info(f"State File: {self.state.state_file}")
        logger.info("")

        completed = self.state.state["completed_stages"]
        if not completed:
            logger.info("No stages completed yet")
        else:
            logger.info("Completed Stages:")
            for stage in completed:
                stage_num = stage.split('_')[1]
                outputs = self.state.get_output(stage)
                logger.info(f"  ✓ Stage {stage_num}: {list(outputs.keys())}")

        logger.info("")
        logger.info("Stored Parameters:")
        params = self.state.state["parameters"]
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 70)


def main():
    """
    Main entry point for InSAR processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="GMTSAR InSAR Time Series Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full sequential pipeline with config file:
  python main.py --project-root /path/to/project --config config.yaml --sequential

  # Run specific stage:
  python main.py --project-root /path/to/project --stage 5

  # Resume from last completed stage:
  python main.py --project-root /path/to/project --resume

  # Show current status:
  python main.py --project-root /path/to/project --status

  # Run stages 5-7 only:
  python main.py --project-root /path/to/project --start-stage 5 --end-stage 7 --sequential
        """
    )

    parser.add_argument(
        '--project-root',
        required=True,
        help='Root directory of the project'
    )

    parser.add_argument(
        '--config',
        help='Configuration file (YAML or JSON)'
    )

    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run all stages sequentially'
    )

    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Run specific stage only'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last completed stage'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current pipeline status'
    )

    parser.add_argument(
        '--start-stage',
        type=int,
        default=1,
        help='Start stage for sequential run'
    )

    parser.add_argument(
        '--end-stage',
        type=int,
        default=10,
        help='End stage for sequential run'
    )

    parser.add_argument(
        '--rerun-from',
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Reset stages from this number onward and re-run (e.g., --rerun-from 6 to re-run stages 6-10)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize pipeline
    pipeline = InSARPipeline(args.project_root, args.config)

    # Execute based on arguments
    try:
        if args.status:
            pipeline.show_status()
        elif args.rerun_from:
            pipeline.reset_from_stage(args.rerun_from)
            end = args.end_stage if args.end_stage != 10 else 10
            pipeline.run_sequential(args.rerun_from, end)
        elif args.resume:
            pipeline.resume()
        elif args.sequential:
            pipeline.run_sequential(args.start_stage, args.end_stage)
        elif args.stage:
            stage_methods = {
                1: pipeline.stage_01_create_project,
                2: pipeline.stage_02_download_data,
                3: pipeline.stage_03_make_dem,
                4: pipeline.stage_04_download_orbits,
                5: pipeline.stage_05_preprocess_subswaths,
                6: pipeline.stage_06_run_interferograms,
                7: pipeline.stage_07_merge,
                8: pipeline.stage_08_unwrap,
                9: pipeline.stage_09_run_sbas,
                10: pipeline.stage_10_post_sbas
            }
            stage_methods[args.stage]()
        else:
            parser.print_help()
            sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
