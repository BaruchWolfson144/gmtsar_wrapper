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
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import processing stages
from make_dir_tree_01 import run_create_project
from stage_02_download_data import run_download_data, validate_download_parameters
from make_dem_03 import run_make_dem
from make_orbits_04 import run_download_orbits
from choose_master_05 import run_preprocess_subswath
from run_interferograms_06 import run_intf
from merge_intfs_07 import run_merge
from unwrap_intfs_08 import run_unwrap

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
        make_dem_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stage 03: Create DEM grid"""
        logger.info("=" * 70)
        logger.info("STAGE 03: Creating DEM Grid")
        logger.info("=" * 70)

        bbox = bbox or self.state.get_parameter("bbox")
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

        logp, result_msg = run_make_dem(self.project_root, bbox_dict, mode, make_dem_path)

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
        username = username or self.state.get_parameter("asf_username")
        password = password or self.state.get_parameter("asf_password")
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
            pin2_lat=pin2_lat
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
        """Stage 05: Choose master and align images"""
        logger.info("=" * 70)
        logger.info("STAGE 05: Preprocessing Subswaths and Choosing Master")
        logger.info("=" * 70)

        orbit = orbit or self.state.get_parameter("orbit", "asc")
        subswath_list = subswath_list or self.state.get_parameter("subswath_list", ["F1", "F2", "F3"])

        # Process each subswath
        results = {}
        master = None

        for sub in subswath_list:
            logger.info(f"Processing subswath {sub}")
            logp = run_preprocess_subswath(self.project_root, orbit, sub)

            # Read the log to extract master information
            if logp and logp.exists():
                with open(logp, 'r') as f:
                    log_data = json.load(f)
                    if not master and "master_selection" in log_data:
                        master = log_data["master_selection"].get("master")
                    results[sub] = {
                        "log_path": str(logp),
                        "master": log_data.get("master_selection", {}).get("master")
                    }

        if not master:
            raise ValueError("Could not determine master image. Check processing logs.")

        outputs = {
            "master": master,
            "subswaths_processed": subswath_list,
            "alignment_complete": True,
            "results": results
        }

        # Store master for future stages
        self.state.set_parameter("master", master)
        self.state.mark_stage_complete("stage_05", outputs)

        logger.info(f"Master image selected: {master}")
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
        threshold_time = threshold_time or self.state.get_parameter("threshold_time", 30)
        threshold_baseline = threshold_baseline or self.state.get_parameter("threshold_baseline", 100)
        config_path = config_path or self.state.get_parameter("config_path")
        subswath_list = subswath_list or self.state.get_parameter("subswath_list", ["F1", "F2", "F3"])

        if not master:
            raise ValueError("Master image not defined. Run stage 05 first or specify manually.")

        if not config_path:
            raise ValueError("config_path is required for stage 06. Provide batch_tops.config file path.")

        results = {}
        for sub in subswath_list:
            logger.info(f"Running interferograms for subswath {sub}")
            logp = run_intf(
                self.project_root,
                orbit,
                sub,
                threshold_time,
                threshold_baseline,
                Path(config_path),
                master
            )
            results[sub] = {"log_path": str(logp) if logp else None}

        outputs = {
            "interferograms_complete": True,
            "subswaths": subswath_list,
            "master": master,
            "threshold_time": threshold_time,
            "threshold_baseline": threshold_baseline,
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
        mode = mode or self.state.get_parameter("merge_mode", 2)

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

        logp, result_msg = run_unwrap(
            self.project_root,
            orbit,
            coherence_threshold=coherence_threshold,
            corr_threshold=corr_threshold,
            max_dis_threshold=max_dis_threshold,
            use_landmask=use_landmask,
            use_mask_def=use_mask_def
        )

        outputs = {
            "unwrap_complete": True,
            "coherence_threshold": coherence_threshold,
            "corr_threshold": corr_threshold,
            "max_dis_threshold": max_dis_threshold,
            "log_path": str(logp) if logp else None
        }

        self.state.mark_stage_complete("stage_08", outputs)
        logger.info(f"Unwrapping completed:\n{result_msg}")
        return outputs

    def run_sequential(self, start_stage: int = 1, end_stage: int = 8):
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
            8: self.stage_08_unwrap
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
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
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
        default=8,
        help='End stage for sequential run'
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
                8: pipeline.stage_08_unwrap
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
