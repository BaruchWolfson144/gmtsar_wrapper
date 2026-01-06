import argparse
import subprocess
import json
from pathlib import Path
import datetime
import shutil
import urllib.parse
import urllib.request
import sys
import glob
import re



def run_cmd(cmd, cwd=None):
    proc = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def prepare_raw_links(project_root: Path, orbit: str, subswath: str):
    """
    Create symbolic links required for GMTSAR Step 5 (raw directory).

    This function links:
    - Sentinel-1 IW{n} VV XML files
    - Sentinel-1 IW{n} VV TIFF files
    - EOF orbit files
    - dem.grd

    Fallback to copy if symlink fails.
    """

    iw_map = {"F1": 1, "F2": 2, "F3": 3}
    iw = iw_map[subswath]

    raw_dir = project_root / orbit / subswath / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def link_many(pattern: str):
        files = glob.glob(pattern)
        if not files:
            raise RuntimeError(f"No files matched: {pattern}")

        for f in files:
            src = Path(f)
            dst = raw_dir / src.name
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)
            except Exception:
                shutil.copy2(src, dst)

    # XML
    link_many(
        str(project_root / orbit / "data" / "F*" / "*.SAFE" / "*" / f"*iw{iw}*vv*xml")
    )

    # TIFF
    link_many(
        str(project_root / orbit / "data" / "F*" / "*.SAFE" / "*" / f"*iw{iw}*vv*tiff")
    )

    # EOF
    link_many(
        str(project_root / orbit / "data" / "*EOF")
    )

    # DEM
    dem_src = project_root / orbit / "topo" / "dem.grd"
    dem_dst = raw_dir / "dem.grd"

    try:
        if dem_dst.exists() or dem_dst.is_symlink():
            dem_dst.unlink()
        dem_dst.symlink_to(dem_src)
    except Exception:
        shutil.copy2(dem_src, dem_dst)


def generate_data_in(raw_dir: Path):
    cmd = "prep_data_linux.csh"
    return(run_cmd(cmd,  cwd=raw_dir))

def run_preproc_mode(raw_dir: Path, mode: int):
    cmd = f"preproc_batch_tops.csh data.in dem.grd {mode} >& pbt_mode{mode}.log"
    return(run_cmd(cmd,  cwd=raw_dir))

def select_master(raw_dir: Path):
    baseline_table = raw_dir / "baseline_table.dat"
    with open(baseline_table) as f:
        rows = [(p[0], float(p[2]), float(p[4])) for p in (line.split() for line in f)]
    time_mean = sum(t[1] for t in rows) / len(rows)
    perp_mean = sum(b[2] for b in rows) / len(rows)
    def dist2(row):
        _, t, b = row
        return (t - time_mean)**2 + (b - perp_mean)**2
    master = min(rows, key=dist2)[0]
    return {
        "master": master,
        "method of choosing": "time+baseline centroid",
        "baseline table lines count": len(rows)
    }

def master_to_first_line(raw_dir: Path, master):
    baseline_table = raw_dir / "baseline_table.dat"
    data_file = raw_dir / "data.in"
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        pattern = re.compile(rf"^.*-{re.escape(master)}")
        master_line = None
        for line in lines:
            if pattern.search(line):
                master_line = line
                break
        if master_line is None:
            raise ValueError(f"Master name '{master}' didn't found in {data_file} file")
        new_lines = [master_line] + [l for l in lines if l != master_line]
    with open(data_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
    return 0
        
        


def write_meta_log(project_root: Path, orbit: str, subswath: str, data_in_info: dict, master_info : dict, preproc_info: list, master_promoted: str):

    meta = {
        "step": 5,
        "orbit": orbit,
        "subswath": subswath,
        "timestamp": datetime.datetime.now().isoformat(),
        "data_in": {"status": data_in_info},
        "master_selection": master_info,
        "preproc": {
            "mode1": preproc_info[0],
            "mode2": preproc_info[1]
        },
        "master_promotion": master_promoted
    }


    logp = (
        project_root
        / "wrapper_meta"
        / "logs"
        / f"step5_{orbit}_{subswath}.json"
    )
    logp.parent.mkdir(parents=True, exist_ok=True)

    with open(logp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return logp



def run_preprocess_subswath(
    project_root: Path,
    orbit: str,
    subswath: str,
):
    raw_dir = project_root / orbit / subswath / "raw" 

    prepare_raw_links(project_root, orbit, subswath)

    rc, _, _ = generate_data_in(raw_dir)
    data_in_info = "ok" if rc == 0 else "failed"

    rc, _, _ = run_preproc_mode(raw_dir, mode=1)
    preproc_1 = "ok" if rc == 0 else "failed"

    master_info = select_master(raw_dir)

    r = master_to_first_line(raw_dir, master_info["master"])
    master_promoted = "ok" if r == 0 else "failed"

    rc, _, _ = run_preproc_mode(raw_dir, mode=2)
    preproc_2 = "ok" if rc == 0 else "failed"

    return(write_meta_log(
        project_root,
        orbit,
        subswath,
        data_in_info,
        master_info,
        [preproc_1, preproc_2],
        master_promoted
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=str, help="path to project root")
    parser.add_argument("orbit", type=str, default="asc", help="asc or dec")
    parser.add_argument("subswath", type=str, default="F1", help="subswath to work on (F1/F2/F3)")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    orbit = args.orbit
    subswath = args.subswath

    run_preprocess_subswath(project_root, orbit, subswath)

if __name__ == "__main__":
    main()
