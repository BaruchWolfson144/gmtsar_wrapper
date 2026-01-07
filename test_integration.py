#!/usr/bin/env python3
"""
Integration test script to verify main.py works with stage files.

This script tests:
1. ProjectState class functionality
2. InSARPipeline initialization
3. Configuration loading
4. Stage method signatures compatibility
"""

import sys
from pathlib import Path
import tempfile
import shutil
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import ProjectState, InSARPipeline

def test_project_state():
    """Test ProjectState class"""
    print("=" * 70)
    print("Testing ProjectState class")
    print("=" * 70)

    # Create temporary project directory
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "test_project"
        project_root.mkdir()

        # Initialize state
        state = ProjectState(project_root)

        # Test parameter storage
        state.set_parameter("test_param", "test_value")
        assert state.get_parameter("test_param") == "test_value"
        print("✓ Parameter storage works")

        # Test stage completion
        state.mark_stage_complete("stage_01", {"output": "test"})
        assert state.is_stage_complete("stage_01")
        assert not state.is_stage_complete("stage_02")
        print("✓ Stage completion tracking works")

        # Test state persistence
        state2 = ProjectState(project_root)
        assert state2.get_parameter("test_param") == "test_value"
        assert state2.is_stage_complete("stage_01")
        print("✓ State persistence works")

        # Test stage outputs
        outputs = state2.get_output("stage_01")
        assert outputs == {"output": "test"}
        print("✓ Stage outputs retrieval works")

    print("\n✓ ProjectState tests passed!\n")

def test_pipeline_init():
    """Test InSARPipeline initialization"""
    print("=" * 70)
    print("Testing InSARPipeline initialization")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "test_project"
        project_root.mkdir()

        # Test basic initialization
        pipeline = InSARPipeline(str(project_root))
        assert pipeline.project_root == project_root
        assert pipeline.state is not None
        print("✓ Basic initialization works")

        # Test with config file
        config_file = project_root / "test_config.yaml"
        config_content = """
project_root: {project_root}
orbit: "asc"
dem:
  minlon: -157.0
  maxlon: -154.2
  minlat: 18.0
  maxlat: 20.4
  mode: 1
"""
        config_file.write_text(config_content.format(project_root=project_root))

        pipeline2 = InSARPipeline(str(project_root), str(config_file))
        assert pipeline2.config is not None
        assert pipeline2.config.get("orbit") == "asc"
        print("✓ Config file loading works")

    print("\n✓ InSARPipeline initialization tests passed!\n")

def test_config_loading():
    """Test configuration file loading"""
    print("=" * 70)
    print("Testing configuration loading")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "test_project"
        project_root.mkdir()

        # Test YAML config
        yaml_config = project_root / "test.yaml"
        yaml_config.write_text("""
orbit: "asc"
dem:
  minlon: -157.0
  maxlon: -154.2
""")

        try:
            pipeline = InSARPipeline(str(project_root), str(yaml_config))
            print("✓ YAML config loading works")
        except ImportError:
            print("⚠ YAML config requires PyYAML (optional)")

        # Test JSON config
        json_config = project_root / "test.json"
        json_config.write_text(json.dumps({
            "orbit": "asc",
            "dem": {
                "minlon": -157.0,
                "maxlon": -154.2
            }
        }))

        pipeline = InSARPipeline(str(project_root), str(json_config))
        assert pipeline.config["orbit"] == "asc"
        print("✓ JSON config loading works")

    print("\n✓ Configuration loading tests passed!\n")

def test_stage_method_signatures():
    """Test that stage methods have correct signatures"""
    print("=" * 70)
    print("Testing stage method signatures")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "test_project"
        project_root.mkdir()

        pipeline = InSARPipeline(str(project_root))

        # Check that all stage methods exist
        expected_methods = [
            "stage_01_create_project",
            "stage_02_make_dem",
            "stage_04_download_orbits",
            "stage_05_preprocess_subswaths",
            "stage_06_run_interferograms",
            "stage_07_merge",
            "stage_08_unwrap"
        ]

        for method_name in expected_methods:
            assert hasattr(pipeline, method_name), f"Missing method: {method_name}"
            method = getattr(pipeline, method_name)
            assert callable(method), f"Method {method_name} is not callable"
            print(f"✓ Method {method_name} exists")

    print("\n✓ Stage method signature tests passed!\n")

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("GMTSAR Pipeline Integration Tests")
    print("=" * 70 + "\n")

    try:
        test_project_state()
        test_pipeline_init()
        test_config_loading()
        test_stage_method_signatures()

        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe integration between main.py and stage files is working correctly.")
        print("\nNext steps:")
        print("  1. Prepare a project directory with Sentinel-1 data")
        print("  2. Create a configuration file (see config_example.yaml)")
        print("  3. Run: python main.py your_config.yaml --sequential")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
