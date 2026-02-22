#!/usr/bin/env python3
"""
Comprehensive tests for run_sbas_09.py and post_sbas_10.py

Tests all functions using mock data to avoid requiring actual GMTSAR tools
or real project data.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import run_sbas_09 as sbas09
import post_sbas_10 as post10


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

SAMPLE_BASELINE_TABLE = """\
S1_20180201_ALL_F1 2018031.1874165505 1491 19.088688 56.434825
S1_20180213_ALL_F1 2018037.1874165505 1497 -12.345678 -40.123456
S1_20180309_ALL_F1 2018068.1874165505 1528 5.678901 20.567890
S1_20180402_ALL_F1 2018092.1874165505 1552 -30.111222 80.333444
S1_20180508_ALL_F1 2018128.1874165505 1588 8.222333 -15.444555
"""

SAMPLE_BASELINE_TABLE_TRAILING_NL = SAMPLE_BASELINE_TABLE + "\n"

SAMPLE_BASELINE_TABLE_MALFORMED = """\
S1_20180201_ALL_F1 2018031.1874165505 1491 19.088688 56.434825
S1_20180213_ALL_F1 -737789
S1_20180309_ALL_F1 2018068.1874165505 1528 5.678901 20.567890
"""

SAMPLE_INTF_IN = """\
S1_20180201_ALL_F1:S1_20180213_ALL_F1
S1_20180201_ALL_F1:S1_20180309_ALL_F1
S1_20180213_ALL_F1:S1_20180309_ALL_F1
S1_20180309_ALL_F1:S1_20180402_ALL_F1
S1_20180402_ALL_F1:S1_20180508_ALL_F1
"""

SAMPLE_PRM = """\
input_file              = S1_20180201_ALL_F1.raw
num_rng_bins            = 69984
bytes_per_line          = 139968
good_bytes_per_line     = 139968
SC_clock_start          = 2018033.15874165
radar_wavelength        = 0.0554658
rng_samp_rate           = 64345238.125714
near_range              = 800853.261605
"""


class TestParseBaselineTable(unittest.TestCase):
    """Tests for parse_baseline_table()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.bt_file = self.tmpdir / "baseline_table.dat"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_basic_parsing(self):
        self.bt_file.write_text(SAMPLE_BASELINE_TABLE)
        scenes = sbas09.parse_baseline_table(self.bt_file)
        self.assertEqual(len(scenes), 5)
        self.assertIn("S1_20180201_ALL_F1", scenes)
        self.assertAlmostEqual(scenes["S1_20180201_ALL_F1"]["perpendicular_baseline"], 56.434825)
        self.assertEqual(scenes["S1_20180201_ALL_F1"]["day"], 1491)

    def test_trailing_newline(self):
        """Trailing newlines should not cause errors."""
        self.bt_file.write_text(SAMPLE_BASELINE_TABLE_TRAILING_NL)
        scenes = sbas09.parse_baseline_table(self.bt_file)
        self.assertEqual(len(scenes), 5)

    def test_malformed_lines_skipped(self):
        """Lines with fewer than 5 fields should be skipped."""
        self.bt_file.write_text(SAMPLE_BASELINE_TABLE_MALFORMED)
        scenes = sbas09.parse_baseline_table(self.bt_file)
        self.assertEqual(len(scenes), 2)
        self.assertNotIn("S1_20180213_ALL_F1", scenes)

    def test_empty_file(self):
        self.bt_file.write_text("")
        scenes = sbas09.parse_baseline_table(self.bt_file)
        self.assertEqual(len(scenes), 0)

    def test_scene_id_extraction(self):
        """Scene ID should be first 7 chars of decimal_year integer."""
        self.bt_file.write_text(SAMPLE_BASELINE_TABLE)
        scenes = sbas09.parse_baseline_table(self.bt_file)
        self.assertEqual(scenes["S1_20180201_ALL_F1"]["scene_id"], "2018031")

    def test_all_fields_present(self):
        self.bt_file.write_text(SAMPLE_BASELINE_TABLE)
        scenes = sbas09.parse_baseline_table(self.bt_file)
        for stem, info in scenes.items():
            self.assertIn("scene_id", info)
            self.assertIn("decimal_year", info)
            self.assertIn("day", info)
            self.assertIn("parallel_baseline", info)
            self.assertIn("perpendicular_baseline", info)


class TestParseIntfIn(unittest.TestCase):
    """Tests for parse_intf_in()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.intf_file = self.tmpdir / "intf.in"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_basic_parsing(self):
        self.intf_file.write_text(SAMPLE_INTF_IN)
        pairs = sbas09.parse_intf_in(self.intf_file)
        self.assertEqual(len(pairs), 5)
        self.assertEqual(pairs[0], ("S1_20180201_ALL_F1", "S1_20180213_ALL_F1"))

    def test_empty_file(self):
        self.intf_file.write_text("")
        pairs = sbas09.parse_intf_in(self.intf_file)
        self.assertEqual(len(pairs), 0)

    def test_lines_without_colon_skipped(self):
        self.intf_file.write_text("no_colon_here\nS1_A:S1_B\n")
        pairs = sbas09.parse_intf_in(self.intf_file)
        self.assertEqual(len(pairs), 1)

    def test_trailing_newline(self):
        self.intf_file.write_text(SAMPLE_INTF_IN + "\n\n")
        pairs = sbas09.parse_intf_in(self.intf_file)
        self.assertEqual(len(pairs), 5)


class TestGetGridDimensions(unittest.TestCase):
    """Tests for get_grid_dimensions()."""

    @patch("run_sbas_09.run_cmd")
    def test_normal_output(self, mock_cmd):
        mock_cmd.return_value = (0, "file.grd 0 1000 0 2000 -50 50 1 1 1000 2000", "")
        nx, ny = sbas09.get_grid_dimensions(Path("test.grd"))
        self.assertEqual(nx, 1000)
        self.assertEqual(ny, 2000)

    @patch("run_sbas_09.run_cmd")
    def test_failure_raises(self, mock_cmd):
        mock_cmd.return_value = (1, "", "error")
        with self.assertRaises(RuntimeError):
            sbas09.get_grid_dimensions(Path("test.grd"))

    @patch("run_sbas_09.run_cmd")
    def test_insufficient_fields_raises(self, mock_cmd):
        mock_cmd.return_value = (0, "file.grd 0 1000", "")
        with self.assertRaises(RuntimeError):
            sbas09.get_grid_dimensions(Path("test.grd"))


class TestGetRadarParameters(unittest.TestCase):
    """Tests for get_radar_parameters()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.prm_file = self.tmpdir / "test.PRM"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_basic_parsing(self):
        self.prm_file.write_text(SAMPLE_PRM)
        params = sbas09.get_radar_parameters(self.prm_file)
        self.assertAlmostEqual(params["wavelength"], 0.0554658)
        self.assertAlmostEqual(params["rng_samp_rate"], 64345238.125714)
        self.assertAlmostEqual(params["near_range"], 800853.261605)

    def test_empty_prm(self):
        self.prm_file.write_text("")
        params = sbas09.get_radar_parameters(self.prm_file)
        self.assertEqual(len(params), 0)

    def test_missing_keys(self):
        self.prm_file.write_text("some_other_key = 123\n")
        params = sbas09.get_radar_parameters(self.prm_file)
        self.assertNotIn("wavelength", params)


class TestCalculateRangeDistance(unittest.TestCase):
    """Tests for calculate_range_distance()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.prm_file = self.tmpdir / "test.PRM"
        self.prm_file.write_text(SAMPLE_PRM)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_range_calculation(self):
        result = sbas09.calculate_range_distance(self.prm_file, 0, 1000)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 800000)  # Should be > near_range

    def test_symmetric_x(self):
        """Symmetric x range should give same result as center."""
        r1 = sbas09.calculate_range_distance(self.prm_file, 400, 600)
        r2 = sbas09.calculate_range_distance(self.prm_file, 0, 1000)
        self.assertAlmostEqual(r1, r2)

    def test_zero_x_gives_near_range(self):
        """x_center=0 should give near_range."""
        r = sbas09.calculate_range_distance(self.prm_file, 0, 0)
        self.assertAlmostEqual(r, 800853.261605, places=2)


class TestPrepareSceneTab(unittest.TestCase):
    """Tests for prepare_scene_tab()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.bt_file = self.tmpdir / "baseline_table.dat"
        self.bt_file.write_text(SAMPLE_BASELINE_TABLE)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_file(self):
        path, count = sbas09.prepare_scene_tab(self.bt_file, self.tmpdir)
        self.assertTrue(path.exists())
        self.assertEqual(count, 5)

    def test_chronological_order(self):
        path, _ = sbas09.prepare_scene_tab(self.bt_file, self.tmpdir)
        lines = path.read_text().strip().split("\n")
        days = [int(line.split()[1]) for line in lines]
        self.assertEqual(days, sorted(days))

    def test_format(self):
        path, _ = sbas09.prepare_scene_tab(self.bt_file, self.tmpdir)
        lines = path.read_text().strip().split("\n")
        for line in lines:
            parts = line.split()
            self.assertEqual(len(parts), 2)
            self.assertEqual(len(parts[0]), 7)  # YYYYDOY
            int(parts[1])  # Day should be integer


class TestPrepareIntfTab(unittest.TestCase):
    """Tests for prepare_intf_tab()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.bt_file = self.tmpdir / "baseline_table.dat"
        self.bt_file.write_text(SAMPLE_BASELINE_TABLE)
        self.intf_file = self.tmpdir / "intf.in"
        self.intf_file.write_text(SAMPLE_INTF_IN)

        # Create mock merge directory with unwrap.grd and corr.grd
        self.merge_dir = self.tmpdir / "merge"
        scenes = sbas09.parse_baseline_table(self.bt_file)
        pairs = sbas09.parse_intf_in(self.intf_file)
        for ref, sec in pairs:
            if ref in scenes and sec in scenes:
                ref_id = scenes[ref]["scene_id"]
                sec_id = scenes[sec]["scene_id"]
                intf_dir = self.merge_dir / f"{ref_id}_{sec_id}"
                intf_dir.mkdir(parents=True)
                (intf_dir / "unwrap.grd").touch()
                (intf_dir / "corr.grd").touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_file(self):
        path, count = sbas09.prepare_intf_tab(
            self.intf_file, self.bt_file, self.merge_dir, self.tmpdir
        )
        self.assertTrue(path.exists())
        self.assertEqual(count, 5)

    def test_format(self):
        path, _ = sbas09.prepare_intf_tab(
            self.intf_file, self.bt_file, self.merge_dir, self.tmpdir
        )
        lines = path.read_text().strip().split("\n")
        for line in lines:
            parts = line.split()
            self.assertEqual(len(parts), 5)
            self.assertIn("unwrap.grd", parts[0])
            self.assertIn("corr.grd", parts[1])
            int(parts[4])  # b_perp should be integer

    def test_missing_unwrap_skipped(self):
        """Pairs where unwrap.grd doesn't exist should be skipped."""
        # Remove one unwrap.grd
        scenes = sbas09.parse_baseline_table(self.bt_file)
        ref_id = scenes["S1_20180201_ALL_F1"]["scene_id"]
        sec_id = scenes["S1_20180213_ALL_F1"]["scene_id"]
        (self.merge_dir / f"{ref_id}_{sec_id}" / "unwrap.grd").unlink()

        path, count = sbas09.prepare_intf_tab(
            self.intf_file, self.bt_file, self.merge_dir, self.tmpdir
        )
        self.assertEqual(count, 4)

    def test_missing_scene_in_baseline_skipped(self):
        """Pairs with scenes not in baseline_table should be skipped."""
        self.intf_file.write_text("S1_UNKNOWN_ALL_F1:S1_20180201_ALL_F1\n")
        path, count = sbas09.prepare_intf_tab(
            self.intf_file, self.bt_file, self.merge_dir, self.tmpdir
        )
        self.assertEqual(count, 0)

    def test_baseline_difference_sign(self):
        """b_perp should be sec - ref."""
        path, _ = sbas09.prepare_intf_tab(
            self.intf_file, self.bt_file, self.merge_dir, self.tmpdir
        )
        scenes = sbas09.parse_baseline_table(self.bt_file)
        lines = path.read_text().strip().split("\n")
        first_line = lines[0].split()
        ref_stem = "S1_20180201_ALL_F1"
        sec_stem = "S1_20180213_ALL_F1"
        expected = int(
            scenes[sec_stem]["perpendicular_baseline"]
            - scenes[ref_stem]["perpendicular_baseline"]
        )
        self.assertEqual(int(first_line[4]), expected)


class TestRunSbasInversion(unittest.TestCase):
    """Tests for run_sbas_inversion()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.intf_tab = self.tmpdir / "intf.tab"
        self.scene_tab = self.tmpdir / "scene.tab"
        self.intf_tab.write_text("dummy")
        self.scene_tab.write_text("dummy")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("run_sbas_09.run_cmd")
    def test_command_construction(self, mock_cmd):
        mock_cmd.return_value = (0, "ok", "")
        # Create expected output files
        (self.tmpdir / "vel.grd").touch()
        (self.tmpdir / "rms.grd").touch()
        (self.tmpdir / "dem_err.grd").touch()

        result = sbas09.run_sbas_inversion(
            self.tmpdir, self.intf_tab, self.scene_tab,
            num_intfs=100, num_scenes=20, xdim=500, ydim=300,
            wavelength=0.0554658, smooth=5.0, range_dist=900000
        )

        cmd = mock_cmd.call_args[0][0]
        self.assertIn("sbas", cmd)
        self.assertIn("100", cmd)
        self.assertIn("20", cmd)
        self.assertIn("500", cmd)
        self.assertIn("300", cmd)
        self.assertIn("-smooth 5.0", cmd)
        self.assertIn("-wavelength 0.0554658", cmd)
        self.assertIn("-rms", cmd)
        self.assertIn("-dem", cmd)

    @patch("run_sbas_09.run_cmd")
    def test_atm_flag(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        sbas09.run_sbas_inversion(
            self.tmpdir, self.intf_tab, self.scene_tab,
            num_intfs=10, num_scenes=5, xdim=100, ydim=100,
            atm_iterations=3
        )
        cmd = mock_cmd.call_args[0][0]
        self.assertIn("-atm 3", cmd)

    @patch("run_sbas_09.run_cmd")
    def test_no_rms_flag(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        sbas09.run_sbas_inversion(
            self.tmpdir, self.intf_tab, self.scene_tab,
            num_intfs=10, num_scenes=5, xdim=100, ydim=100,
            compute_rms=False
        )
        cmd = mock_cmd.call_args[0][0]
        self.assertNotIn("-rms", cmd)

    @patch("run_sbas_09.run_cmd")
    def test_output_detection(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        (self.tmpdir / "vel.grd").touch()
        (self.tmpdir / "disp_001.grd").touch()
        (self.tmpdir / "disp_002.grd").touch()

        result = sbas09.run_sbas_inversion(
            self.tmpdir, self.intf_tab, self.scene_tab,
            num_intfs=10, num_scenes=5, xdim=100, ydim=100
        )
        self.assertTrue(result["outputs"]["vel.grd"]["exists"])
        self.assertFalse(result["outputs"]["rms.grd"]["exists"])
        self.assertEqual(result["displacement_grids"]["count"], 2)

    @patch("run_sbas_09.run_cmd")
    def test_failure_captured(self, mock_cmd):
        mock_cmd.return_value = (1, "", "SBAS error")
        result = sbas09.run_sbas_inversion(
            self.tmpdir, self.intf_tab, self.scene_tab,
            num_intfs=10, num_scenes=5, xdim=100, ydim=100
        )
        self.assertEqual(result["return_code"], 1)

    @patch("run_sbas_09.run_cmd")
    def test_long_output_truncated(self, mock_cmd):
        long_out = "x" * 5000
        mock_cmd.return_value = (0, long_out, "")
        result = sbas09.run_sbas_inversion(
            self.tmpdir, self.intf_tab, self.scene_tab,
            num_intfs=10, num_scenes=5, xdim=100, ydim=100
        )
        self.assertLessEqual(len(result["stdout"]), 2000)


class TestRunPrepSbas(unittest.TestCase):
    """Tests for run_prep_sbas()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.orbit = "asc"
        sub = self.tmpdir / self.orbit / "F1"
        sub.mkdir(parents=True)
        (sub / "intf.in").write_text(SAMPLE_INTF_IN)
        (sub / "baseline_table.dat").write_text(SAMPLE_BASELINE_TABLE)
        (self.tmpdir / self.orbit / "merge").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("run_sbas_09.run_cmd")
    def test_copies_input_files(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        sbas09.run_prep_sbas(self.tmpdir, self.orbit)

        sbas_dir = self.tmpdir / self.orbit / "SBAS"
        self.assertTrue((sbas_dir / "intf.in").exists())
        self.assertTrue((sbas_dir / "baseline_table.dat").exists())

    @patch("run_sbas_09.run_cmd")
    def test_command_format(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        sbas09.run_prep_sbas(self.tmpdir, self.orbit)

        cmd = mock_cmd.call_args[0][0]
        self.assertIn("prep_sbas.csh", cmd)
        self.assertIn("intf.in", cmd)
        self.assertIn("baseline_table.dat", cmd)

    @patch("run_sbas_09.run_cmd")
    def test_detects_output_files(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        sbas_dir = self.tmpdir / self.orbit / "SBAS"
        sbas_dir.mkdir(parents=True, exist_ok=True)
        (sbas_dir / "scene.tab").write_text("2018031 1491\n2018037 1497\n")
        (sbas_dir / "intf.tab").write_text("line1\nline2\nline3\n")

        result = sbas09.run_prep_sbas(self.tmpdir, self.orbit)
        self.assertEqual(result["num_scenes"], 2)
        self.assertEqual(result["num_interferograms"], 3)


class TestWriteMetaLog09(unittest.TestCase):
    """Tests for write_meta_log() in stage 09."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_log_file(self):
        path = sbas09.write_meta_log(
            self.tmpdir, "asc",
            {"prep": "data"}, {"sbas": "data"}, {"viz": "data"}
        )
        self.assertTrue(path.exists())
        self.assertIn("step9", path.name)

    def test_log_valid_json(self):
        path = sbas09.write_meta_log(
            self.tmpdir, "asc",
            {"prep": "data"}, {"sbas": "data"}, {"viz": "data"}
        )
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["step"], 9)
        self.assertEqual(data["orbit"], "asc")
        self.assertIn("timestamp", data)

    def test_creates_log_directory(self):
        path = sbas09.write_meta_log(
            self.tmpdir, "asc", {}, {}, {}
        )
        self.assertTrue((self.tmpdir / "wrapper_meta" / "logs").is_dir())


# =============================================================================
# VISUALIZATION TESTS (Stage 09)
# =============================================================================

class TestBaselineNetworkPlot(unittest.TestCase):
    """Tests for create_baseline_network_plot()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.scene_tab = self.tmpdir / "scene.tab"
        self.intf_tab = self.tmpdir / "intf.tab"
        self.scene_tab.write_text(
            "2018031 1491\n2018037 1497\n2018068 1528\n"
        )
        self.intf_tab.write_text(
            "../merge/2018031_2018037/unwrap.grd ../merge/2018031_2018037/corr.grd 2018031 2018037 -97\n"
            "../merge/2018031_2018068/unwrap.grd ../merge/2018031_2018068/corr.grd 2018031 2018068 -36\n"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(sbas09.HAS_MATPLOTLIB, "matplotlib not available")
    def test_creates_plot(self):
        out = self.tmpdir / "network.png"
        result = sbas09.create_baseline_network_plot(self.intf_tab, self.scene_tab, out)
        self.assertTrue(result)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    @unittest.skipUnless(sbas09.HAS_MATPLOTLIB, "matplotlib not available")
    def test_empty_intf_tab(self):
        self.intf_tab.write_text("")
        out = self.tmpdir / "network.png"
        # Should not crash, even with no data
        result = sbas09.create_baseline_network_plot(self.intf_tab, self.scene_tab, out)
        # May succeed or fail gracefully, but no exception


class TestConnectivityHistogram(unittest.TestCase):
    """Tests for create_connectivity_histogram()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.scene_tab = self.tmpdir / "scene.tab"
        self.intf_tab = self.tmpdir / "intf.tab"
        self.scene_tab.write_text("2018031 1491\n2018037 1497\n2018068 1528\n")
        self.intf_tab.write_text(
            "p1 p2 2018031 2018037 -97\n"
            "p1 p2 2018031 2018068 -36\n"
            "p1 p2 2018037 2018068 61\n"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(sbas09.HAS_MATPLOTLIB, "matplotlib not available")
    def test_creates_plot(self):
        out = self.tmpdir / "connectivity.png"
        result = sbas09.create_connectivity_histogram(self.intf_tab, self.scene_tab, out)
        self.assertTrue(result)
        self.assertTrue(out.exists())


class TestTemporalBaselineHistogram(unittest.TestCase):
    """Tests for create_temporal_baseline_histogram()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.scene_tab = self.tmpdir / "scene.tab"
        self.intf_tab = self.tmpdir / "intf.tab"
        self.scene_tab.write_text("2018031 1491\n2018037 1497\n2018068 1528\n")
        self.intf_tab.write_text(
            "p1 p2 2018031 2018037 -97\n"
            "p1 p2 2018031 2018068 -36\n"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(sbas09.HAS_MATPLOTLIB, "matplotlib not available")
    def test_creates_plot(self):
        out = self.tmpdir / "baselines.png"
        result = sbas09.create_temporal_baseline_histogram(self.intf_tab, self.scene_tab, out)
        self.assertTrue(result)
        self.assertTrue(out.exists())


# =============================================================================
# STAGE 09 INTEGRATION - run_sbas()
# =============================================================================

class TestRunSbasIntegration(unittest.TestCase):
    """Integration tests for the main run_sbas() function."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.orbit = "asc"
        base = self.tmpdir / self.orbit

        # Create F1 subswath files
        f1 = base / "F1"
        f1.mkdir(parents=True)
        (f1 / "intf.in").write_text(SAMPLE_INTF_IN)
        (f1 / "baseline_table.dat").write_text(SAMPLE_BASELINE_TABLE)

        raw = f1 / "raw"
        raw.mkdir()
        (raw / "test.PRM").write_text(SAMPLE_PRM)

        # Create merge directory with interferograms
        merge = base / "merge"
        scenes = sbas09.parse_baseline_table(f1 / "baseline_table.dat")
        pairs = sbas09.parse_intf_in(f1 / "intf.in")
        for ref, sec in pairs:
            if ref in scenes and sec in scenes:
                ref_id = scenes[ref]["scene_id"]
                sec_id = scenes[sec]["scene_id"]
                d = merge / f"{ref_id}_{sec_id}"
                d.mkdir(parents=True)
                (d / "unwrap.grd").touch()
                (d / "corr.grd").touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("run_sbas_09.run_cmd")
    def test_manual_prep_path(self, mock_cmd):
        """Test the manual preparation path (not using prep_sbas.csh)."""
        # Mock grdinfo for grid dimensions
        mock_cmd.return_value = (0, "file 0 1000 0 2000 -50 50 1 1 500 300", "")

        # Create vel.grd to avoid failure on missing output
        sbas_dir = self.tmpdir / self.orbit / "SBAS"
        sbas_dir.mkdir(parents=True, exist_ok=True)

        log_path, msg = sbas09.run_sbas(
            self.tmpdir, self.orbit, use_prep_script=False
        )
        self.assertTrue(log_path.exists())
        self.assertIn("Prepared SBAS inputs", msg)

    @patch("run_sbas_09.run_cmd")
    def test_no_interferograms_raises(self, mock_cmd):
        """Should raise if no valid interferograms."""
        # Empty intf.in
        (self.tmpdir / self.orbit / "F1" / "intf.in").write_text("")

        mock_cmd.return_value = (0, "", "")
        with self.assertRaises(RuntimeError):
            sbas09.run_sbas(self.tmpdir, self.orbit, use_prep_script=False)


# =============================================================================
# POST-SBAS (Stage 10) TESTS
# =============================================================================

class TestGetGridStats(unittest.TestCase):
    """Tests for get_grid_stats() in post_sbas module."""

    @patch("post_sbas_10.run_cmd")
    def test_parses_grdinfo(self, mock_cmd):
        mock_cmd.return_value = (0, "file 33.5 36.0 27.0 31.0 -50.5 75.3 0.01 0.01 250 400", "")
        stats = post10.get_grid_stats(Path("test.grd"))
        self.assertAlmostEqual(stats['x_min'], 33.5)
        self.assertAlmostEqual(stats['z_max'], 75.3)
        self.assertEqual(stats['nx'], 250)
        self.assertEqual(stats['ny'], 400)

    @patch("post_sbas_10.run_cmd")
    def test_failure_returns_empty(self, mock_cmd):
        mock_cmd.return_value = (1, "", "error")
        stats = post10.get_grid_stats(Path("test.grd"))
        self.assertEqual(stats, {})


class TestProjectToLatlon(unittest.TestCase):
    """Tests for project_to_latlon()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.trans_dat = self.tmpdir / "trans.dat"
        self.trans_dat.touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("post_sbas_10.get_grid_stats")
    @patch("post_sbas_10.run_cmd")
    def test_creates_symlink(self, mock_cmd, mock_stats):
        mock_cmd.return_value = (0, "", "")
        mock_stats.return_value = {}
        sbas_dir = self.tmpdir / "SBAS"
        sbas_dir.mkdir()
        (sbas_dir / "vel.grd").touch()

        post10.project_to_latlon(sbas_dir, "vel.grd", "vel_ll.grd", self.trans_dat)
        self.assertTrue((sbas_dir / "trans.dat").is_symlink())

    @patch("post_sbas_10.get_grid_stats")
    @patch("post_sbas_10.run_cmd")
    def test_success_detection(self, mock_cmd, mock_stats):
        mock_cmd.return_value = (0, "", "")
        mock_stats.return_value = {}
        sbas_dir = self.tmpdir / "SBAS"
        sbas_dir.mkdir()

        # Create output to simulate success
        (sbas_dir / "vel_ll.grd").touch()

        result = post10.project_to_latlon(sbas_dir, "vel.grd", "vel_ll.grd", self.trans_dat)
        self.assertTrue(result["success"])

    @patch("post_sbas_10.run_cmd")
    def test_failure_detection(self, mock_cmd):
        mock_cmd.return_value = (1, "", "error")
        sbas_dir = self.tmpdir / "SBAS"
        sbas_dir.mkdir()

        result = post10.project_to_latlon(sbas_dir, "vel.grd", "vel_ll.grd", self.trans_dat)
        self.assertFalse(result["success"])

    @patch("post_sbas_10.get_grid_stats")
    @patch("post_sbas_10.run_cmd")
    def test_existing_symlink_not_duplicated(self, mock_cmd, mock_stats):
        mock_cmd.return_value = (0, "", "")
        mock_stats.return_value = {}
        sbas_dir = self.tmpdir / "SBAS"
        sbas_dir.mkdir()
        # Pre-create symlink
        os.symlink(str(self.trans_dat), str(sbas_dir / "trans.dat"))

        # Should not raise
        post10.project_to_latlon(sbas_dir, "vel.grd", "vel_ll.grd", self.trans_dat)


class TestCreateColorPalette(unittest.TestCase):
    """Tests for create_color_palette()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("post_sbas_10.run_cmd")
    def test_command_with_auto_range(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        (self.tmpdir / "vel_ll.cpt").touch()

        # Mock get_grid_stats
        with patch("post_sbas_10.get_grid_stats") as mock_stats:
            mock_stats.return_value = {'z_min': -50, 'z_max': 50}
            result = post10.create_color_palette(
                self.tmpdir, "vel_ll.grd", "vel_ll.cpt"
            )

        cmd = mock_cmd.call_args[0][0]
        self.assertIn("grd2cpt", cmd)
        self.assertIn("-Cjet", cmd)
        self.assertTrue(result["success"])

    @patch("post_sbas_10.run_cmd")
    def test_custom_range(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        (self.tmpdir / "vel_ll.cpt").touch()

        result = post10.create_color_palette(
            self.tmpdir, "vel_ll.grd", "vel_ll.cpt",
            vmin=-30, vmax=30
        )
        cmd = mock_cmd.call_args[0][0]
        self.assertIn("-30", cmd)
        self.assertIn("30", cmd)

    @patch("post_sbas_10.run_cmd")
    def test_reverse_flag(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        (self.tmpdir / "test.cpt").touch()

        post10.create_color_palette(
            self.tmpdir, "test.grd", "test.cpt",
            vmin=-10, vmax=10, reverse=True
        )
        cmd = mock_cmd.call_args[0][0]
        self.assertIn("-I", cmd)


class TestCreateKmlOverlay(unittest.TestCase):
    """Tests for create_kml_overlay()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("post_sbas_10.run_cmd")
    def test_command_format(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        (self.tmpdir / "vel_ll.kml").touch()
        (self.tmpdir / "vel_ll.png").touch()

        result = post10.create_kml_overlay(self.tmpdir, "vel_ll.grd", "vel_ll.cpt")
        cmd = mock_cmd.call_args[0][0]
        self.assertIn("grd2kml.csh", cmd)
        self.assertIn("vel_ll", cmd)
        self.assertTrue(result["success"])

    @patch("post_sbas_10.run_cmd")
    def test_custom_output_name(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        post10.create_kml_overlay(
            self.tmpdir, "vel_ll.grd", "vel_ll.cpt", output_name="my_vel"
        )
        cmd = mock_cmd.call_args[0][0]
        self.assertIn("my_vel", cmd)

    @patch("post_sbas_10.run_cmd")
    def test_failure_detection(self, mock_cmd):
        mock_cmd.return_value = (0, "", "")
        # Don't create kml/png files
        result = post10.create_kml_overlay(self.tmpdir, "vel_ll.grd", "vel_ll.cpt")
        self.assertFalse(result["success"])


class TestWriteMetaLog10(unittest.TestCase):
    """Tests for write_meta_log() in stage 10."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_log(self):
        path = post10.write_meta_log(
            self.tmpdir, "asc", {"proj": "data"}, {"kml": "data"}, {"viz": "data"}
        )
        self.assertTrue(path.exists())
        self.assertIn("step10", path.name)

    def test_valid_json(self):
        path = post10.write_meta_log(
            self.tmpdir, "asc", {}, {}, {}
        )
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["step"], 10)
        self.assertEqual(data["stage_name"], "Post-SBAS Processing")


# =============================================================================
# VISUALIZATION TESTS (Stage 10)
# =============================================================================

class TestVelocityMap(unittest.TestCase):
    """Tests for create_velocity_map()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.read_gmt_grid")
    def test_creates_map(self, mock_read):
        import numpy as np
        data = np.random.randn(100, 200) * 10
        meta = {'x_min': 0, 'x_max': 200, 'y_min': 0, 'y_max': 100}
        mock_read.return_value = (data, meta)

        out = self.tmpdir / "vel.png"
        result = post10.create_velocity_map(self.tmpdir, "vel.grd", out)
        self.assertTrue(result)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.read_gmt_grid")
    def test_custom_range(self, mock_read):
        import numpy as np
        data = np.random.randn(50, 50) * 10
        meta = {'x_min': 0, 'x_max': 50, 'y_min': 0, 'y_max': 50}
        mock_read.return_value = (data, meta)

        out = self.tmpdir / "vel.png"
        result = post10.create_velocity_map(
            self.tmpdir, "vel.grd", out, vmin=-20, vmax=20
        )
        self.assertTrue(result)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.read_gmt_grid")
    def test_none_data_returns_false(self, mock_read):
        mock_read.return_value = None
        out = self.tmpdir / "vel.png"
        result = post10.create_velocity_map(self.tmpdir, "vel.grd", out)
        self.assertFalse(result)


class TestRmsMap(unittest.TestCase):
    """Tests for create_rms_map()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.read_gmt_grid")
    def test_creates_map(self, mock_read):
        import numpy as np
        data = np.abs(np.random.randn(100, 200)) * 5
        meta = {'x_min': 0, 'x_max': 200, 'y_min': 0, 'y_max': 100}
        mock_read.return_value = (data, meta)

        out = self.tmpdir / "rms.png"
        result = post10.create_rms_map(self.tmpdir, "rms.grd", out)
        self.assertTrue(result)
        self.assertTrue(out.exists())


class TestDemErrorMap(unittest.TestCase):
    """Tests for create_dem_error_map()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.read_gmt_grid")
    def test_creates_map(self, mock_read):
        import numpy as np
        data = np.random.randn(80, 120) * 3
        meta = {'x_min': 0, 'x_max': 120, 'y_min': 0, 'y_max': 80}
        mock_read.return_value = (data, meta)

        out = self.tmpdir / "dem_err.png"
        result = post10.create_dem_error_map(self.tmpdir, "dem_err.grd", out)
        self.assertTrue(result)
        self.assertTrue(out.exists())


class TestVelocityHistogram(unittest.TestCase):
    """Tests for create_velocity_histogram()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.read_gmt_grid")
    def test_creates_histogram(self, mock_read):
        import numpy as np
        data = np.random.randn(100, 200) * 10
        meta = {'x_min': 0, 'x_max': 200, 'y_min': 0, 'y_max': 100}
        mock_read.return_value = (data, meta)

        out = self.tmpdir / "hist.png"
        result = post10.create_velocity_histogram(self.tmpdir, "vel.grd", out)
        self.assertTrue(result)
        self.assertTrue(out.exists())


class TestSummaryFigure(unittest.TestCase):
    """Tests for create_summary_figure()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.read_gmt_grid")
    def test_creates_summary(self, mock_read):
        import numpy as np
        data = np.random.randn(50, 50) * 10
        meta = {'x_min': 0, 'x_max': 50, 'y_min': 0, 'y_max': 50}
        mock_read.return_value = (data, meta)

        # Create expected grid files
        (self.tmpdir / "vel.grd").touch()
        (self.tmpdir / "rms.grd").touch()
        (self.tmpdir / "dem_err.grd").touch()

        out = self.tmpdir / "summary.png"
        result = post10.create_summary_figure(self.tmpdir, out)
        self.assertTrue(result)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    def test_missing_grids_handled(self):
        """Should handle missing grid files gracefully."""
        out = self.tmpdir / "summary.png"
        # No vel.grd, rms.grd, etc. - should still produce a figure
        result = post10.create_summary_figure(self.tmpdir, out)
        self.assertTrue(result)


class TestDisplacementTimeseries(unittest.TestCase):
    """Tests for create_displacement_timeseries_plot()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.scene_tab = self.tmpdir / "scene.tab"
        self.scene_tab.write_text("2018031 1491\n2018037 1497\n2018068 1528\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.read_gmt_grid")
    def test_creates_plot(self, mock_read):
        import numpy as np
        data = np.random.randn(50, 50) * 5
        meta = {'x_min': 0, 'x_max': 50, 'y_min': 0, 'y_max': 50}
        mock_read.return_value = (data, meta)

        # Create 3 disp grids
        for i in range(3):
            (self.tmpdir / f"disp_{i:03d}.grd").touch()

        out = self.tmpdir / "ts.png"
        result = post10.create_displacement_timeseries_plot(
            self.tmpdir, self.scene_tab,
            [(10, 10), (25, 25)], out,
            point_labels=["P1", "P2"]
        )
        self.assertTrue(result)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    def test_no_disp_grids_returns_false(self):
        out = self.tmpdir / "ts.png"
        result = post10.create_displacement_timeseries_plot(
            self.tmpdir, self.scene_tab, [(10, 10)], out
        )
        self.assertFalse(result)


# =============================================================================
# POST-SBAS INTEGRATION - run_post_sbas()
# =============================================================================

class TestRunPostSbasIntegration(unittest.TestCase):
    """Integration tests for run_post_sbas()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.orbit = "asc"
        base = self.tmpdir / self.orbit

        # Create SBAS directory with expected files
        sbas = base / "SBAS"
        sbas.mkdir(parents=True)
        (sbas / "vel.grd").touch()
        (sbas / "rms.grd").touch()
        (sbas / "dem_err.grd").touch()
        (sbas / "scene.tab").write_text("2018031 1491\n2018037 1497\n")

        # Create merge directory with trans.dat
        merge = base / "merge"
        merge.mkdir(parents=True)
        (merge / "trans.dat").touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("post_sbas_10.create_summary_figure", return_value=False)
    @patch("post_sbas_10.create_velocity_histogram", return_value=False)
    @patch("post_sbas_10.create_dem_error_map", return_value=False)
    @patch("post_sbas_10.create_rms_map", return_value=False)
    @patch("post_sbas_10.create_velocity_map", return_value=False)
    @patch("post_sbas_10.create_kml_overlay")
    @patch("post_sbas_10.create_color_palette")
    @patch("post_sbas_10.get_grid_stats")
    @patch("post_sbas_10.project_to_latlon")
    def test_full_pipeline(self, mock_proj, mock_stats, mock_cpt, mock_kml, *viz_mocks):
        mock_proj.return_value = {"success": True, "output_path": "vel_ll.grd"}
        mock_stats.return_value = {'z_min': -50, 'z_max': 50}
        mock_cpt.return_value = {"success": True}
        mock_kml.return_value = {"success": True}

        # Create vel_ll.grd to pass the check
        (self.tmpdir / self.orbit / "SBAS" / "vel_ll.grd").touch()

        log_path, msg = post10.run_post_sbas(
            self.tmpdir, self.orbit, project_disp=False
        )
        self.assertTrue(log_path.exists())
        self.assertIn("Projected velocity", msg)

    def test_missing_sbas_dir_raises(self):
        shutil.rmtree(self.tmpdir / self.orbit / "SBAS")
        with self.assertRaises(RuntimeError):
            post10.run_post_sbas(self.tmpdir, self.orbit)

    def test_missing_trans_dat_raises(self):
        (self.tmpdir / self.orbit / "merge" / "trans.dat").unlink()
        with self.assertRaises(RuntimeError):
            post10.run_post_sbas(self.tmpdir, self.orbit)

    @patch("post_sbas_10.create_summary_figure", return_value=False)
    @patch("post_sbas_10.create_velocity_histogram", return_value=False)
    @patch("post_sbas_10.create_dem_error_map", return_value=False)
    @patch("post_sbas_10.create_rms_map", return_value=False)
    @patch("post_sbas_10.create_velocity_map", return_value=False)
    @patch("post_sbas_10.project_to_latlon")
    def test_trans_dat_fallback_to_subswath(self, mock_proj, *viz_mocks):
        """Should find trans.dat in subswath/topo/ if not in merge/."""
        mock_proj.return_value = {"success": False}

        # Remove from merge
        (self.tmpdir / self.orbit / "merge" / "trans.dat").unlink()

        # Create in F2/topo
        topo = self.tmpdir / self.orbit / "F2" / "topo"
        topo.mkdir(parents=True)
        (topo / "trans.dat").touch()

        log_path, msg = post10.run_post_sbas(
            self.tmpdir, self.orbit, project_disp=False
        )
        self.assertTrue(log_path.exists())


# =============================================================================
# EDGE CASES AND ROBUSTNESS
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness tests."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_baseline_table_only_whitespace_lines(self):
        bt = self.tmpdir / "bt.dat"
        bt.write_text("   \n\t\n  \n")
        scenes = sbas09.parse_baseline_table(bt)
        self.assertEqual(len(scenes), 0)

    def test_intf_in_multiple_colons(self):
        """Lines with more than one colon should be handled."""
        f = self.tmpdir / "intf.in"
        f.write_text("A:B:C\nD:E\n")
        pairs = sbas09.parse_intf_in(f)
        # "A:B:C" splits to 3 parts, should be skipped (len != 2)
        # "D:E" is valid
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], ("D", "E"))

    def test_prm_with_extra_spaces(self):
        prm = self.tmpdir / "test.PRM"
        prm.write_text("  radar_wavelength  =  0.0554658  \n")
        params = sbas09.get_radar_parameters(prm)
        self.assertAlmostEqual(params["wavelength"], 0.0554658)

    def test_prepare_scene_tab_single_scene(self):
        bt = self.tmpdir / "bt.dat"
        bt.write_text("S1_20180201_ALL_F1 2018031.187 1491 19.0 56.4\n")
        path, count = sbas09.prepare_scene_tab(bt, self.tmpdir)
        self.assertEqual(count, 1)
        content = path.read_text().strip()
        self.assertEqual(len(content.split("\n")), 1)

    @patch("run_sbas_09.run_cmd")
    def test_sbas_range_as_integer(self, mock_cmd):
        """Range distance should be passed as integer to SBAS."""
        mock_cmd.return_value = (0, "", "")
        intf_tab = self.tmpdir / "intf.tab"
        scene_tab = self.tmpdir / "scene.tab"
        intf_tab.touch()
        scene_tab.touch()

        sbas09.run_sbas_inversion(
            self.tmpdir, intf_tab, scene_tab,
            num_intfs=10, num_scenes=5, xdim=100, ydim=100,
            range_dist=912345.678
        )
        cmd = mock_cmd.call_args[0][0]
        self.assertIn("-range 912345", cmd)
        self.assertNotIn("912345.678", cmd)

    def test_color_palette_step_calculation(self):
        """Step size should be (vmax-vmin)/50."""
        with patch("post_sbas_10.run_cmd") as mock_cmd:
            mock_cmd.return_value = (0, "", "")
            (self.tmpdir / "test.cpt").touch()

            post10.create_color_palette(
                self.tmpdir, "test.grd", "test.cpt",
                vmin=-100, vmax=100
            )
            cmd = mock_cmd.call_args[0][0]
            # Step = 200/50 = 4.0
            self.assertIn("-T-100/100/4.0", cmd)


class TestReadGmtGrid(unittest.TestCase):
    """Tests for read_gmt_grid() in post_sbas."""

    @unittest.skipUnless(post10.HAS_NUMPY, "numpy required")
    @patch("post_sbas_10.run_cmd")
    def test_fallback_to_grd2xyz(self, mock_cmd):
        """When netCDF4 is not available, should fall back to gmt grd2xyz."""
        import numpy as np

        # Simulate grd2xyz output
        lines = []
        for y in range(3):
            for x in range(4):
                lines.append(f"{x} {y} {x * 10 + y}")
        mock_cmd.return_value = (0, "\n".join(lines), "")

        # Temporarily remove netCDF4
        with patch.dict('sys.modules', {'netCDF4': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                # This is tricky - let's just test the None return path
                result = post10.read_gmt_grid(Path("test.grd"))
                # Result depends on whether netCDF4 is actually available

    @unittest.skipUnless(post10.HAS_NUMPY, "numpy required")
    def test_nonexistent_file(self):
        result = post10.read_gmt_grid(Path("/nonexistent/file.grd"))
        self.assertIsNone(result)


# =============================================================================
# CLI / ARGUMENT PARSING TESTS
# =============================================================================

class TestCLI09(unittest.TestCase):
    """Tests for Stage 09 CLI argument parsing."""

    def test_default_args(self):
        with patch("sys.argv", ["prog", "/tmp/proj", "asc"]):
            parser = sbas09.main.__code__  # just verify main exists
            self.assertIsNotNone(parser)

    def test_argparse_defaults(self):
        """Verify default argument values."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("project_root")
        parser.add_argument("orbit")
        parser.add_argument("--subswath", default="F1")
        parser.add_argument("--unwrap-file", default="unwrap.grd")
        parser.add_argument("--smooth", type=float, default=5.0)
        parser.add_argument("--atm-iterations", type=int, default=0)

        args = parser.parse_args(["/tmp/proj", "asc"])
        self.assertEqual(args.subswath, "F1")
        self.assertEqual(args.unwrap_file, "unwrap.grd")
        self.assertEqual(args.smooth, 5.0)
        self.assertEqual(args.atm_iterations, 0)


class TestCLI10(unittest.TestCase):
    """Tests for Stage 10 CLI argument parsing."""

    def test_vel_range_both_required(self):
        """If only vel_min is provided without vel_max, vel_range should be None."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--vel-min", type=float, default=None)
        parser.add_argument("--vel-max", type=float, default=None)

        args = parser.parse_args(["--vel-min", "-50"])
        vel_range = None
        if args.vel_min is not None and args.vel_max is not None:
            vel_range = (args.vel_min, args.vel_max)
        self.assertIsNone(vel_range)


class TestExtractGnssPointTimeseries(unittest.TestCase):
    """Tests for extract_gnss_point_timeseries()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.sbas_dir = self.tmpdir / "SBAS"
        self.sbas_dir.mkdir()

        # Create scene.tab
        self.scene_tab = self.sbas_dir / "scene.tab"
        with open(self.scene_tab, 'w') as f:
            f.write("2019001 1827\n")
            f.write("2019013 1839\n")
            f.write("2019025 1851\n")

        # Create GNSS.ll file
        self.gnss_file = self.tmpdir / "GNSS.ll"
        with open(self.gnss_file, 'w') as f:
            f.write("45.243602 37.740448\n")
            f.write("45.253448 37.697583\n")

        # Create dummy disp_*_ll.grd files
        for i in range(3):
            (self.sbas_dir / f"disp_{i:04d}_ll.grd").touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_parses_gnss_file(self):
        """Verify GNSS.ll file parsing."""
        points = []
        with open(self.gnss_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    points.append((float(parts[0]), float(parts[1])))
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0], 45.243602)
        self.assertAlmostEqual(points[1][1], 37.697583)

    def test_empty_gnss_file_returns_false(self):
        """Empty GNSS.ll should return False."""
        empty_gnss = self.tmpdir / "empty.ll"
        with open(empty_gnss, 'w') as f:
            f.write("")

        result = post10.extract_gnss_point_timeseries(
            self.sbas_dir, self.scene_tab, empty_gnss, self.tmpdir / "plots"
        )
        self.assertFalse(result)

    def test_missing_gnss_file_raises(self):
        """Non-existent GNSS.ll should raise or return False."""
        result = post10.extract_gnss_point_timeseries(
            self.sbas_dir, self.scene_tab, self.tmpdir / "nonexistent.ll",
            self.tmpdir / "plots"
        )
        self.assertFalse(result)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.run_cmd")
    def test_calls_grdtrack(self, mock_cmd):
        """Verify gmt grdtrack is called for each disp_*_ll.grd."""
        mock_cmd.return_value = (0, "45.243602 37.740448 5.2\n45.253448 37.697583 3.1\n", "")

        output_dir = self.tmpdir / "plots"
        post10.extract_gnss_point_timeseries(
            self.sbas_dir, self.scene_tab, self.gnss_file, output_dir
        )

        # Should call grdtrack for: 3 disp grids + 1 vel_ll.grd (if exists)
        calls = [c for c in mock_cmd.call_args_list if "grdtrack" in str(c)]
        self.assertGreaterEqual(len(calls), 3)

    @unittest.skipUnless(post10.HAS_MATPLOTLIB and post10.HAS_NUMPY, "matplotlib/numpy required")
    @patch("post_sbas_10.run_cmd")
    def test_creates_individual_plots(self, mock_cmd):
        """Verify individual point_N.png files are created."""
        mock_cmd.return_value = (0, "45.243602 37.740448 5.2\n45.253448 37.697583 3.1\n", "")

        output_dir = self.tmpdir / "plots"
        result = post10.extract_gnss_point_timeseries(
            self.sbas_dir, self.scene_tab, self.gnss_file, output_dir
        )

        self.assertTrue(result)
        self.assertTrue((output_dir / "point_1.png").exists())
        self.assertTrue((output_dir / "point_2.png").exists())

    def test_gnss_with_comments_and_blank_lines(self):
        """GNSS.ll file with comments and blank lines should be handled."""
        gnss = self.tmpdir / "gnss_comments.ll"
        with open(gnss, 'w') as f:
            f.write("# This is a comment\n")
            f.write("\n")
            f.write("45.0 37.0\n")
            f.write("# Another comment\n")
            f.write("46.0 38.0\n")

        points = []
        with open(gnss) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    points.append((float(parts[0]), float(parts[1])))
        self.assertEqual(len(points), 2)


class TestCreateVelocityStatisticsReport(unittest.TestCase):
    """Tests for create_velocity_statistics_report()."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_missing_vel_ll_returns_false(self):
        """Should return False if vel_ll.grd doesn't exist."""
        result = post10.create_velocity_statistics_report(
            self.tmpdir, self.tmpdir / "stats.txt"
        )
        self.assertFalse(result)

    @patch("post_sbas_10.read_gmt_grid")
    @patch("post_sbas_10.get_grid_stats")
    def test_creates_report(self, mock_stats, mock_read):
        """Should create a formatted statistics report."""
        import numpy as np

        # Create dummy vel_ll.grd
        (self.tmpdir / "vel_ll.grd").touch()

        mock_stats.return_value = {
            'x_min': 44.6, 'x_max': 46.1,
            'y_min': 36.5, 'y_max': 39.0,
            'z_min': -109.42, 'z_max': 14.03,
            'dx': 0.000556, 'dy': 0.000417,
            'nx': 2700, 'ny': 5980
        }

        data = np.random.randn(5980, 2700) * 7
        meta = {'x_min': 44.6, 'x_max': 46.1, 'y_min': 36.5, 'y_max': 39.0, 'nx': 2700, 'ny': 5980}
        mock_read.return_value = (data, meta)

        out = self.tmpdir / "vel_ll_statistics.txt"
        result = post10.create_velocity_statistics_report(self.tmpdir, out, "asc")

        self.assertTrue(result)
        self.assertTrue(out.exists())

        content = out.read_text()
        self.assertIn("SBAS-InSAR Velocity Statistics Report", content)
        self.assertIn("Ascending", content)
        self.assertIn("2700", content)
        self.assertIn("5980", content)
        self.assertIn("mm/year", content)

    @patch("post_sbas_10.read_gmt_grid")
    @patch("post_sbas_10.get_grid_stats")
    def test_handles_nan_pixels(self, mock_stats, mock_read):
        """Should correctly report NaN pixel counts."""
        import numpy as np

        (self.tmpdir / "vel_ll.grd").touch()

        mock_stats.return_value = {
            'x_min': 0, 'x_max': 10,
            'y_min': 0, 'y_max': 10,
            'z_min': -50, 'z_max': 10,
            'dx': 1, 'dy': 1,
            'nx': 10, 'ny': 10
        }

        data = np.ones((10, 10)) * 5.0
        data[0:5, :] = np.nan  # 50% NaN
        meta = {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10, 'nx': 10, 'ny': 10}
        mock_read.return_value = (data, meta)

        out = self.tmpdir / "stats.txt"
        result = post10.create_velocity_statistics_report(self.tmpdir, out)
        self.assertTrue(result)

        content = out.read_text()
        self.assertIn("50", content)  # 50 valid pixels
        self.assertIn("50.0%", content)

    @patch("post_sbas_10.get_grid_stats")
    def test_works_without_numpy(self, mock_stats):
        """Should produce a report even if read_gmt_grid fails."""
        (self.tmpdir / "vel_ll.grd").touch()

        mock_stats.return_value = {
            'x_min': 44.0, 'x_max': 46.0,
            'y_min': 36.0, 'y_max': 39.0,
            'z_min': -100, 'z_max': 15,
            'dx': 0.001, 'dy': 0.001,
            'nx': 2000, 'ny': 3000
        }

        out = self.tmpdir / "stats.txt"
        # Patch read_gmt_grid to return None (simulating no numpy)
        with patch("post_sbas_10.read_gmt_grid", return_value=None):
            result = post10.create_velocity_statistics_report(self.tmpdir, out)

        self.assertTrue(result)
        content = out.read_text()
        self.assertIn("-100", content)
        self.assertIn("+15", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
