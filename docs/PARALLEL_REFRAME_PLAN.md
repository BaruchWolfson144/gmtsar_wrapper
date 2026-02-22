# תוכנית מקבול שלב 4: Reframing

**תאריך יצירה:** 2026-02-04
**סטטוס:** תוכנית לשימוש עתידי (לא מיושם)
**ענף מוצע:** `experimental/parallel-reframe`

---

## תוכן עניינים

1. [רקע והקשר](#1-רקע-והקשר)
2. [ניתוח הסקריפט הנוכחי](#2-ניתוח-הסקריפט-הנוכחי)
3. [למה מקבול לא טריוויאלי](#3-למה-מקבול-לא-טריוויאלי)
4. [אסטרטגיה נבחרת: בידוד תיקיות](#4-אסטרטגיה-נבחרת-בידוד-תיקיות)
5. [ארכיטקטורה מפורטת](#5-ארכיטקטורה-מפורטת)
6. [קוד מלא למימוש](#6-קוד-מלא-למימוש)
7. [באגים פוטנציאליים ופתרונות](#7-באגים-פוטנציאליים-ופתרונות)
8. [בדיקות נדרשות לפני מימוש](#8-בדיקות-נדרשות-לפני-מימוש)
9. [אינטגרציה עם הwrapper](#9-אינטגרציה-עם-הwrapper)
10. [הערכת ביצועים](#10-הערכת-ביצועים)

---

## 1. רקע והקשר

### מהו Reframing?

Reframing הוא תהליך של **חיתוך ואיחוד** פריימים של Sentinel-1 TOPS SLC לאזור גיאוגרפי מוגדר.

**למה צריך את זה:**
- Sentinel-1 מצלם ב"פריימים" קבועים (~250 ק"מ אורך)
- לפעמים אזור העניין חוצה גבול בין שני פריימים
- צריך "לתפור" את הפריימים יחד ולחתוך לאזור הרצוי

**בפרויקט שלנו (Track 58):**
- 122 תאריכי רכישה
- 2 פריימים לכל תאריך (F0440 + F0473 או דומה)
- סה"כ 244 קבצי SAFE
- פינים: 32.5°E, 27.7°N עד 32.5°E, 29.65°N

### המצב הנוכחי

שלב 4 כולל:
1. **הורדת אורביטים** (~5 דקות) - מהיר, נשאר סדרתי
2. **Reframing** (~35 דקות) - צוואר בקבוק, מטרת המקבול

**הסקריפט האחראי:** `/usr/local/GMTSAR_master/bin/organize_files_tops_linux.csh`

---

## 2. ניתוח הסקריפט הנוכחי

### חתימת הסקריפט

```bash
organize_files_tops_linux.csh <SAFE_filelist> <pins.ll> <mode>
```

| פרמטר | תיאור |
|-------|-------|
| `SAFE_filelist` | קובץ טקסט עם נתיבים מלאים לתיקיות SAFE |
| `pins.ll` | קובץ עם 2 שורות: `lon1 lat1` ו-`lon2 lat2` |
| `mode` | 1=הכנה+אורביטים, 2=reframing בפועל |

### מה הסקריפט עושה (mode 2)

```
לכל תיקיית SAFE ברשימה:
    1. קרא metadata (manifest.safe, annotation/*.xml)
    2. חשב footprint של הסצנה
    3. המר pins.ll לקואורדינטות רדאר (SAT_llt2rat)
    4. בדוק אם הפינים בתוך ה-footprint
    5. אם כן:
        - חתוך את ה-bursts הרלוונטיים
        - צור תיקיית SAFE חדשה עם הנתונים החתוכים
        - שמור ב-F{start}_F{end}/
    6. אם לא:
        - דלג (הודעת warning)
```

### קבצי פלט

```
data/
├── SAFE_filelist                    # קלט - רשימת כל ה-SAFE
├── pins.ll                          # קלט - קואורדינטות הפינים
├── S1A_IW_SLC__1SDV_20141009...SAFE # קלט - SAFE מקורי
├── S1A_IW_SLC__1SDV_20141021...SAFE # קלט - SAFE מקורי
├── ...
└── F0440_F0473/                     # פלט - SAFE מרופרמים
    ├── S1A_IW_SLC__1SDV_20141009...SAFE
    ├── S1A_IW_SLC__1SDV_20141021...SAFE
    └── ...
```

### קבצים זמניים שהסקריפט יוצר

בזמן הריצה, הסקריפט יוצר:
- `tmp*` - קבצים זמניים שונים
- `new.SAFE` - תיקייה זמנית למבנה SAFE החדש
- `topo/` - אם נדרש חישוב טופוגרפיה

**חשוב:** שמות הקבצים הזמניים **קבועים** - לא כוללים PID או timestamp!

---

## 3. למה מקבול לא טריוויאלי

### בעיה 1: הסקריפט מעבד הכל ביחד

```csh
# מתוך organize_files_tops_linux.csh, שורות ~100-150
foreach safe (`cat $1`)
    # עיבוד $safe
end
```

הסקריפט לא מקבל פרמטר "עבד רק תאריך X" - הוא תמיד עובר על כל הרשימה.

### בעיה 2: קבצים זמניים עם שמות קבועים

```csh
# דוגמאות מהסקריפט
set tmp1 = "tmp_ll2ra"
set tmp2 = "tmp_burst"
mkdir new.SAFE
```

אם שני תהליכים רצים **באותה תיקייה** במקביל:
- Worker A כותב ל-`tmp_ll2ra`
- Worker B דורס את `tmp_ll2ra`
- Worker A קורא נתונים שגויים
- **תוצאה: נתונים פגומים!**

### בעיה 3: תיקיית פלט משותפת

כל התאריכים נכתבים לאותה `F0440_F0473/`. אין בעיה של דריסה (כל SAFE בשם ייחודי), אבל:
- יכולות להיות בעיות permissions אם שני workers מנסים ליצור את התיקייה
- אין אפשרות לזהות איזה worker יצר איזה קובץ

### למה זה **כן** ניתן למקבול לוגית

**נקודת מפתח:** הריפריימינג של תאריך A **לא תלוי בכלל** בתאריך B.

כל תאריך:
- קורא רק את ה-SAFE files שלו (2 קבצים)
- משתמש באותם פינים (read-only)
- יוצר SAFE חדש רק לעצמו

**מסקנה:** הבעיה היא טכנית (איך הסקריפט כתוב), לא לוגית.

---

## 4. אסטרטגיה נבחרת: בידוד תיקיות

### הרעיון המרכזי

**במקום** לשנות את סקריפט GMTSAR, נריץ אותו **בתיקיות נפרדות** לכל תאריך.

```
/tmp/reframe_20141009_12345/    # Worker 1
    ├── data/
    │   ├── S1A...20141009...SAFE -> /original/path (symlink)
    │   └── S1A...20141009...SAFE -> /original/path (symlink)
    ├── SAFE_filelist
    ├── pins.ll
    └── F0440_F0473/            # פלט
        └── S1A...20141009...SAFE

/tmp/reframe_20141021_12346/    # Worker 2
    ├── data/
    │   ├── S1A...20141021...SAFE -> /original/path (symlink)
    │   └── S1A...20141021...SAFE -> /original/path (symlink)
    ├── SAFE_filelist
    ├── pins.ll
    └── F0440_F0473/            # פלט
        └── S1A...20141021...SAFE
```

### יתרונות

| יתרון | הסבר |
|-------|------|
| **אין שינוי ב-GMTSAR** | לא צריך fork או תחזוקה |
| **בידוד מלא** | כל worker בתיקייה משלו, אין התנגשויות |
| **ניהול שגיאות קל** | תאריך נכשל? האחרים ממשיכים |
| **קל לדיבוג** | אפשר לשמור את התיקיות הזמניות לבדיקה |

### חסרונות

| חסרון | פתרון |
|--------|-------|
| Overhead של symlinks | זניח (~1ms לתאריך) |
| צריך merge בסוף | פונקציה פשוטה, ~1 שנייה |
| מקום ב-/tmp | משתמשים ב-symlinks, לא העתקה |

---

## 5. ארכיטקטורה מפורטת

### תרשים זרימה

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Main Process                                 │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 1: Preparation                                          │  │
│  │   1. Read SAFE_filelist                                       │  │
│  │   2. Group by date: {20141009: [safe1, safe2], ...}          │  │
│  │   3. Read pins.ll content                                     │  │
│  │   4. Create work_items list                                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 2: Parallel Execution                                   │  │
│  │   ProcessPoolExecutor(max_workers=N)                         │  │
│  │   Submit: reframe_single_date(work_item) for each date       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │              │              │              │               │
│         ▼              ▼              ▼              ▼               │
│    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐            │
│    │Worker 1│    │Worker 2│    │Worker 3│    │Worker 4│            │
│    │20141009│    │20141021│    │20141102│    │20141114│            │
│    └────────┘    └────────┘    └────────┘    └────────┘            │
│         │              │              │              │               │
│         └──────────────┴──────────────┴──────────────┘               │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 3: Merge Results                                        │  │
│  │   1. Collect all successful results                          │  │
│  │   2. Move SAFE files to final F*_F*/ directory              │  │
│  │   3. Verify count                                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Phase 4: Cleanup                                              │  │
│  │   Delete all temp directories                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Worker Flow (פירוט)

```
reframe_single_date(work_item)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Create temp directory            │
│    /tmp/reframe_{date}_{pid}/       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Create data/ subdirectory        │
│    /tmp/reframe_{date}_{pid}/data/  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Create symlinks to SAFE files    │
│    data/S1A...SAFE -> original      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Write SAFE_filelist              │
│    (paths to symlinks)              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Write pins.ll                    │
│    (copy from main)                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. cd to temp directory             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 7. Run organize_files_tops_linux.csh│
│    SAFE_filelist pins.ll 2          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 8. Find output F*_F*/ directory     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 9. Return result dict               │
│    {date, success, output_dir, ...} │
└─────────────────────────────────────┘
```

---

## 6. קוד מלא למימוש

### קובץ חדש: `parallel_reframe.py`

```python
#!/usr/bin/env python3
"""
parallel_reframe.py - Parallel reframing for GMTSAR TOPS processing

This module provides parallel reframing capability for Stage 4 of the
GMTSAR wrapper pipeline. It uses directory isolation to run multiple
instances of organize_files_tops_linux.csh simultaneously.

Usage:
    # As module
    from parallel_reframe import run_parallel_reframe
    success, results = run_parallel_reframe(project_root, orbit, num_workers=4)

    # As standalone (for testing)
    python3 parallel_reframe.py /path/to/project asc --workers 4

Author: Claude (with human guidance)
Date: 2026-02-04
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Timeout for single date reframing (seconds)
REFRAME_TIMEOUT = 600  # 10 minutes per date should be plenty

# Temporary directory base
TEMP_BASE = Path("/tmp")

# GMTSAR script name
REFRAME_SCRIPT = "organize_files_tops_linux.csh"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_safe_dates(safe_filelist: Path) -> dict[str, list[str]]:
    """
    Group SAFE file paths by acquisition date.

    Args:
        safe_filelist: Path to SAFE_filelist file

    Returns:
        Dictionary mapping date strings to list of SAFE paths.
        Example: {'20141009': ['/path/to/S1A...20141009...SAFE', ...]}
    """
    dates_dict = defaultdict(list)

    with open(safe_filelist) as f:
        for line in f:
            safe_path = line.strip()
            if not safe_path:
                continue

            # Extract date from SAFE filename
            # Format: S1A_IW_SLC__1SDV_20141009T155542_...
            match = re.search(r'_(\d{8})T', safe_path)
            if match:
                date = match.group(1)
                dates_dict[date].append(safe_path)
            else:
                print(f"Warning: Could not extract date from {safe_path}")

    return dict(dates_dict)


def validate_work_item(work_item: dict) -> tuple[bool, str]:
    """
    Validate that a work item has all required data.

    Args:
        work_item: Dictionary with date, safe_paths, pins_content, data_dir

    Returns:
        (is_valid, error_message)
    """
    required_keys = ['date', 'safe_paths', 'pins_content', 'data_dir']
    for key in required_keys:
        if key not in work_item:
            return False, f"Missing required key: {key}"

    if not work_item['safe_paths']:
        return False, "No SAFE paths provided"

    if not work_item['pins_content'].strip():
        return False, "Empty pins content"

    # Check that SAFE files exist
    for safe_path in work_item['safe_paths']:
        if not Path(safe_path).exists():
            return False, f"SAFE file not found: {safe_path}"

    return True, ""


# =============================================================================
# WORKER FUNCTION
# =============================================================================

def reframe_single_date(work_item: dict) -> dict:
    """
    Worker function: reframe a single date in an isolated directory.

    This function is executed in a separate process by ProcessPoolExecutor.
    It creates a temporary workspace, sets up symlinks, runs the GMTSAR
    reframing script, and returns the results.

    Args:
        work_item: Dictionary containing:
            - date: Date string (YYYYMMDD)
            - safe_paths: List of full paths to original SAFE directories
            - pins_content: Content of pins.ll file
            - data_dir: Original data directory path

    Returns:
        Dictionary containing:
            - date: Date string
            - success: Boolean
            - output_dir: Path to F*_F* directory (if successful)
            - safe_count: Number of SAFE files created (if successful)
            - frame_range: Name of F*_F* directory (if successful)
            - duration: Processing time in seconds
            - error: Error message (if failed)
            - stdout: Script stdout (truncated)
            - stderr: Script stderr (truncated)
    """
    date = work_item['date']
    safe_paths = work_item['safe_paths']
    pins_content = work_item['pins_content']

    start_time = time.time()
    result = {
        'date': date,
        'success': False,
        'duration': 0,
    }

    # Validate input
    is_valid, error_msg = validate_work_item(work_item)
    if not is_valid:
        result['error'] = f"Validation failed: {error_msg}"
        result['duration'] = time.time() - start_time
        return result

    # ==========================================================================
    # Step 1: Create isolated temporary directory
    # ==========================================================================
    # Include PID to ensure uniqueness even if same date is retried
    temp_base = TEMP_BASE / f"reframe_{date}_{os.getpid()}"
    temp_data = temp_base / "data"

    try:
        temp_data.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        result['error'] = f"Failed to create temp directory: {e}"
        result['duration'] = time.time() - start_time
        return result

    try:
        # ======================================================================
        # Step 2: Create symlinks to original SAFE directories
        # ======================================================================
        temp_safe_paths = []
        for orig_path in safe_paths:
            orig_safe = Path(orig_path)
            link_path = temp_data / orig_safe.name

            try:
                link_path.symlink_to(orig_safe.resolve())
                temp_safe_paths.append(str(link_path))
            except FileExistsError:
                # Symlink already exists (shouldn't happen with unique temp dirs)
                temp_safe_paths.append(str(link_path))
            except Exception as e:
                result['error'] = f"Failed to create symlink for {orig_safe.name}: {e}"
                result['duration'] = time.time() - start_time
                return result

        # ======================================================================
        # Step 3: Create SAFE_filelist for this date only
        # ======================================================================
        temp_filelist = temp_base / "SAFE_filelist"
        try:
            temp_filelist.write_text('\n'.join(temp_safe_paths) + '\n')
        except Exception as e:
            result['error'] = f"Failed to write SAFE_filelist: {e}"
            result['duration'] = time.time() - start_time
            return result

        # ======================================================================
        # Step 4: Create pins.ll (same for all dates)
        # ======================================================================
        temp_pins = temp_base / "pins.ll"
        try:
            temp_pins.write_text(pins_content)
        except Exception as e:
            result['error'] = f"Failed to write pins.ll: {e}"
            result['duration'] = time.time() - start_time
            return result

        # ======================================================================
        # Step 5: Change to temp directory and run script
        # ======================================================================
        original_dir = os.getcwd()

        try:
            os.chdir(temp_base)

            cmd = f"{REFRAME_SCRIPT} {temp_filelist} {temp_pins} 2"

            proc_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=REFRAME_TIMEOUT
            )

            result['stdout'] = proc_result.stdout[-1000:] if proc_result.stdout else ""
            result['stderr'] = proc_result.stderr[-1000:] if proc_result.stderr else ""

            if proc_result.returncode != 0:
                result['error'] = f"Script returned {proc_result.returncode}"
                result['duration'] = time.time() - start_time
                return result

        except subprocess.TimeoutExpired:
            result['error'] = f"Timeout after {REFRAME_TIMEOUT}s"
            result['duration'] = time.time() - start_time
            return result

        except Exception as e:
            result['error'] = f"Script execution failed: {e}"
            result['duration'] = time.time() - start_time
            return result

        finally:
            os.chdir(original_dir)

        # ======================================================================
        # Step 6: Find output directory
        # ======================================================================
        output_dirs = list(temp_base.glob("F*_F*"))

        if not output_dirs:
            # Check if pins were outside the scene
            if "outside" in result.get('stderr', '').lower():
                result['error'] = "Pins outside scene footprint"
            else:
                result['error'] = "No F*_F* directory created"
            result['duration'] = time.time() - start_time
            return result

        output_dir = output_dirs[0]

        # ======================================================================
        # Step 7: Count created SAFE files
        # ======================================================================
        created_safes = list(output_dir.glob("*.SAFE"))

        if not created_safes:
            result['error'] = "F*_F* directory is empty"
            result['duration'] = time.time() - start_time
            return result

        # ======================================================================
        # Success!
        # ======================================================================
        result['success'] = True
        result['output_dir'] = str(output_dir)
        result['safe_count'] = len(created_safes)
        result['frame_range'] = output_dir.name
        result['duration'] = time.time() - start_time

        return result

    except Exception as e:
        result['error'] = f"Unexpected error: {e}"
        result['duration'] = time.time() - start_time
        return result

    # Note: We don't clean up temp_base here!
    # The main process needs to move the files first.


# =============================================================================
# ORCHESTRATION FUNCTIONS
# =============================================================================

def prepare_work_items(
    project_root: Path,
    orbit: str
) -> tuple[list[dict], str]:
    """
    Prepare work items for parallel reframing.

    Args:
        project_root: Project root directory
        orbit: Orbit type (asc/des)

    Returns:
        (list of work items, pins_content string)
    """
    data_dir = project_root / orbit / "data"
    safe_filelist = data_dir / "SAFE_filelist"
    pins_file = project_root / orbit / "reframed" / "pins.ll"

    # Read pins.ll
    if not pins_file.exists():
        raise FileNotFoundError(f"pins.ll not found at {pins_file}")
    pins_content = pins_file.read_text()

    # Validate pins content
    lines = [l.strip() for l in pins_content.strip().split('\n') if l.strip()]
    if len(lines) != 2:
        raise ValueError(f"pins.ll must have exactly 2 lines, found {len(lines)}")

    # Read and group SAFE files
    if not safe_filelist.exists():
        raise FileNotFoundError(f"SAFE_filelist not found at {safe_filelist}")

    dates_dict = parse_safe_dates(safe_filelist)

    if not dates_dict:
        raise ValueError("No valid SAFE files found in SAFE_filelist")

    # Create work items
    work_items = []
    for date, safe_paths in sorted(dates_dict.items()):
        work_items.append({
            'date': date,
            'safe_paths': safe_paths,
            'pins_content': pins_content,
            'data_dir': str(data_dir),
        })

    return work_items, pins_content


def merge_results(
    project_root: Path,
    orbit: str,
    results: list[dict]
) -> tuple[bool, int, list[str]]:
    """
    Merge all reframed SAFE files into the final F*_F* directory.

    Args:
        project_root: Project root directory
        orbit: Orbit type (asc/des)
        results: List of result dicts from workers

    Returns:
        (success, moved_count, list of errors)
    """
    data_dir = project_root / orbit / "data"
    errors = []

    # Find all unique frame ranges
    frame_ranges = set()
    for r in results:
        if r['success'] and 'frame_range' in r:
            frame_ranges.add(r['frame_range'])

    if not frame_ranges:
        return False, 0, ["No successful reframes to merge"]

    # If multiple frame ranges, warn and use the most common
    if len(frame_ranges) > 1:
        # Count occurrences
        range_counts = defaultdict(int)
        for r in results:
            if r['success'] and 'frame_range' in r:
                range_counts[r['frame_range']] += 1

        target_frame = max(range_counts, key=range_counts.get)
        errors.append(
            f"Multiple frame ranges found: {frame_ranges}. "
            f"Using most common: {target_frame}"
        )
    else:
        target_frame = list(frame_ranges)[0]

    # Create target directory
    target_dir = data_dir / target_frame
    target_dir.mkdir(exist_ok=True)

    # Move all SAFE files
    moved_count = 0
    for r in results:
        if not r['success']:
            continue

        output_dir = Path(r['output_dir'])
        if not output_dir.exists():
            errors.append(f"Output directory missing for {r['date']}: {output_dir}")
            continue

        for safe_dir in output_dir.glob("*.SAFE"):
            dest = target_dir / safe_dir.name

            if dest.exists():
                errors.append(f"Skipped {safe_dir.name}: already exists")
                continue

            try:
                shutil.move(str(safe_dir), str(dest))
                moved_count += 1
            except Exception as e:
                errors.append(f"Failed to move {safe_dir.name}: {e}")

    # Verify final count
    final_count = len(list(target_dir.glob("*.SAFE")))
    expected = sum(r.get('safe_count', 0) for r in results if r['success'])

    if final_count != expected:
        errors.append(f"Count mismatch: expected {expected}, found {final_count}")

    success = (moved_count > 0 and len(errors) == 0)
    return success, moved_count, errors


def cleanup_temp_directories(results: list[dict]) -> int:
    """
    Remove all temporary directories created during parallel reframing.

    Args:
        results: List of result dicts from workers

    Returns:
        Number of directories cleaned up
    """
    cleaned = 0

    for r in results:
        if 'output_dir' not in r:
            continue

        output_dir = Path(r['output_dir'])
        temp_base = output_dir.parent  # /tmp/reframe_{date}_{pid}

        if temp_base.exists() and temp_base.name.startswith('reframe_'):
            try:
                shutil.rmtree(temp_base)
                cleaned += 1
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_base}: {e}")

    return cleaned


def run_parallel_reframe(
    project_root: Path,
    orbit: str,
    num_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> tuple[bool, list[dict]]:
    """
    Run reframing in parallel for all dates.

    This is the main entry point for parallel reframing.

    Args:
        project_root: Project root directory
        orbit: Orbit type (asc/des)
        num_workers: Number of parallel workers (default: 4)
        progress_callback: Optional callback(completed, total, result) for progress

    Returns:
        (overall_success, list of result dicts)
    """
    print(f"\n{'='*60}")
    print(f"PARALLEL REFRAMING")
    print(f"{'='*60}")

    # ==========================================================================
    # Phase 1: Preparation
    # ==========================================================================
    print(f"\n[Phase 1] Preparing work items...")

    try:
        work_items, pins_content = prepare_work_items(project_root, orbit)
    except Exception as e:
        print(f"ERROR: Preparation failed: {e}")
        return False, []

    total_dates = len(work_items)
    print(f"  Found {total_dates} dates to process")
    print(f"  Using {num_workers} parallel workers")

    # ==========================================================================
    # Phase 2: Parallel Execution
    # ==========================================================================
    print(f"\n[Phase 2] Running parallel reframing...")

    results = []
    completed = 0
    failed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(reframe_single_date, item): item['date']
            for item in work_items
        }

        # Process results as they complete
        for future in as_completed(futures):
            date = futures[future]

            try:
                result = future.result(timeout=REFRAME_TIMEOUT + 60)
                results.append(result)

                completed += 1
                if result['success']:
                    status = f"OK ({result['safe_count']} SAFE, {result['duration']:.1f}s)"
                    symbol = "✓"
                else:
                    failed += 1
                    status = f"FAILED: {result.get('error', 'Unknown')[:40]}"
                    symbol = "✗"

                progress = f"[{completed}/{total_dates}]"
                print(f"  {symbol} {progress} {date}: {status}")

                if progress_callback:
                    progress_callback(completed, total_dates, result)

            except Exception as e:
                failed += 1
                results.append({
                    'date': date,
                    'success': False,
                    'error': f"Future exception: {e}",
                    'duration': 0
                })
                print(f"  ✗ [{completed}/{total_dates}] {date}: EXCEPTION: {e}")

    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Success: {completed - failed}/{total_dates}, Failed: {failed}")

    # ==========================================================================
    # Phase 3: Merge Results
    # ==========================================================================
    print(f"\n[Phase 3] Merging results...")

    merge_success, moved_count, merge_errors = merge_results(
        project_root, orbit, results
    )

    print(f"  Moved {moved_count} SAFE files")
    if merge_errors:
        for err in merge_errors[:5]:  # Show first 5 errors
            print(f"  Warning: {err}")
        if len(merge_errors) > 5:
            print(f"  ... and {len(merge_errors) - 5} more warnings")

    # ==========================================================================
    # Phase 4: Cleanup
    # ==========================================================================
    print(f"\n[Phase 4] Cleaning up temporary directories...")

    cleaned = cleanup_temp_directories(results)
    print(f"  Removed {cleaned} temp directories")

    # ==========================================================================
    # Summary
    # ==========================================================================
    overall_success = (failed == 0 and merge_success)

    print(f"\n{'='*60}")
    if overall_success:
        print(f"PARALLEL REFRAMING COMPLETED SUCCESSFULLY")
    else:
        print(f"PARALLEL REFRAMING COMPLETED WITH ERRORS")
    print(f"{'='*60}\n")

    return overall_success, results


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel reframing for GMTSAR TOPS processing"
    )
    parser.add_argument(
        "project_root",
        help="Project root directory"
    )
    parser.add_argument(
        "orbit",
        choices=["asc", "des"],
        help="Orbit type"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, don't execute"
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()

    if args.dry_run:
        print("DRY RUN - showing work items only")
        work_items, pins = prepare_work_items(project_root, args.orbit)
        print(f"\nWould process {len(work_items)} dates:")
        for item in work_items[:5]:
            print(f"  {item['date']}: {len(item['safe_paths'])} SAFE files")
        if len(work_items) > 5:
            print(f"  ... and {len(work_items) - 5} more")
        return

    success, results = run_parallel_reframe(
        project_root,
        args.orbit,
        num_workers=args.workers
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

---

## 7. באגים פוטנציאליים ופתרונות

### באג 1: תלות בקבצי EOF (אורביטים)

**הבעיה:** הסקריפט `organize_files_tops_linux.csh` אולי צריך גישה לקבצי EOF לחישוב המסלול המדויק.

**בדיקה נדרשת:**
```bash
grep -n "EOF\|orbit\|\.EOF" /usr/local/GMTSAR_master/bin/organize_files_tops_linux.csh
```

**פתרון אם נדרש:**
```python
# בתוך reframe_single_date, לפני הרצת הסקריפט:
orbit_dir = Path(work_item['data_dir']).parent / "orbit"
temp_orbit = temp_base / "orbit"
temp_orbit.mkdir(exist_ok=True)

for eof_file in orbit_dir.glob("*.EOF"):
    link_path = temp_orbit / eof_file.name
    link_path.symlink_to(eof_file.resolve())
```

**סטטוס:** לבדוק לפני מימוש!

---

### באג 2: נתיבים יחסיים בסקריפט

**הבעיה:** אם הסקריפט עושה `realpath` או `readlink -f` על הקלט, הוא יקבל את הנתיב המקורי במקום הsymlink.

**מה לבדוק:**
```bash
grep -n "realpath\|readlink\|pwd" /usr/local/GMTSAR_master/bin/organize_files_tops_linux.csh
```

**פתרון אפשרי:**
במקום symlinks, להשתמש ב-hard links (אם באותה מערכת קבצים) או העתקה מלאה (יקר יותר).

---

### באג 3: Frame ranges שונים בין תאריכים

**הבעיה:** ESA שינתה את מספרי הפריימים לאורך המשימה. תאריך מ-2014 יכול ליצור `F0440_F0470`, ותאריך מ-2024 יכול ליצור `F0440_F0473`.

**איך לזהות:**
```bash
# אחרי ריצה מוצלחת, בדוק:
ls /tmp/reframe_*/F*_F* | awk -F'/' '{print $NF}' | sort | uniq -c
```

**פתרון מומלץ:** הקוד כבר מטפל בזה - הוא משתמש ב-frame range הנפוץ ביותר ומעביר את כל הקבצים לשם.

---

### באג 4: Symlinks שבורים

**הבעיה:** אם הנתיב המקורי משתנה או הדיסק לא mounted, הsymlinks יהיו שבורים.

**פתרון:** הוספת בדיקה ב-`validate_work_item`:
```python
for safe_path in work_item['safe_paths']:
    p = Path(safe_path)
    if not p.exists():
        return False, f"SAFE file not found: {safe_path}"
    if p.is_symlink() and not p.resolve().exists():
        return False, f"Broken symlink: {safe_path}"
```

---

### באג 5: מקום בדיסק ב-/tmp

**הבעיה:** /tmp יכול להיות partition קטן או tmpfs בזיכרון.

**פתרון:**
1. לבדוק מקום פנוי לפני הריצה
2. לאפשר קונפיגורציה של TEMP_BASE
3. לנקות תיקיות של workers שסיימו ברגע שה-merge נעשה

```python
# בתחילת run_parallel_reframe:
import shutil
total, used, free = shutil.disk_usage(TEMP_BASE)
required = total_dates * 10 * 1024 * 1024  # ~10MB per date estimate
if free < required:
    raise RuntimeError(f"Not enough space in {TEMP_BASE}: {free/1e9:.1f}GB free, need ~{required/1e9:.1f}GB")
```

---

### באג 6: הרשאות

**הבעיה:** הסקריפט GMTSAR יכול ליצור קבצים עם הרשאות שונות מהצפוי.

**פתרון:** אחרי ה-merge, לוודא הרשאות:
```python
for safe_dir in target_dir.glob("*.SAFE"):
    safe_dir.chmod(0o755)
    for f in safe_dir.rglob("*"):
        if f.is_file():
            f.chmod(0o644)
        elif f.is_dir():
            f.chmod(0o755)
```

---

### באג 7: תאריכים עם פריים בודד

**הבעיה:** אם יש תאריך עם רק פריים אחד (לא 2), הריפריימינג לא אמור לרוץ עליו.

**פתרון:** לסנן בהכנה:
```python
# ב-prepare_work_items:
for date, safe_paths in sorted(dates_dict.items()):
    if len(safe_paths) < 2:
        print(f"  Warning: Skipping {date} - only {len(safe_paths)} SAFE file(s)")
        continue
    work_items.append(...)
```

---

## 8. בדיקות נדרשות לפני מימוש

### בדיקה 1: תלות ב-EOF

```bash
# בדוק אם הסקריפט מתייחס לקבצי אורביט:
grep -n "\.EOF\|orbit\|precise\|restituted" \
    /usr/local/GMTSAR_master/bin/organize_files_tops_linux.csh
```

### בדיקה 2: קבצים זמניים

```bash
# הרץ את הסקריפט על תאריך בודד וראה מה נוצר:
cd /tmp/test_reframe
organize_files_tops_linux.csh test_SAFE_filelist pins.ll 2
ls -la
```

### בדיקה 3: ריצה ידנית של שני תאריכים במקביל

```bash
# Terminal 1:
cd /tmp/reframe_test1
organize_files_tops_linux.csh SAFE_filelist pins.ll 2

# Terminal 2 (simultaneously):
cd /tmp/reframe_test2
organize_files_tops_linux.csh SAFE_filelist pins.ll 2

# בדוק שהתוצאות תקינות בשניהם
```

### בדיקה 4: Frame range consistency

```bash
# בדוק אם יש הבדלים בפלט בין תאריכים:
for date in 20141009 20180315 20241015; do
    echo "=== $date ==="
    ls -d /tmp/reframe_${date}*/F*_F*
done
```

---

## 9. אינטגרציה עם ה-Wrapper

### שינויים נדרשים ב-make_orbits_04.py

```python
# הוסף import בראש הקובץ:
from parallel_reframe import run_parallel_reframe

# ב-download_orbits_with_reframe(), החלף את mode 2 בריצה מקבילית:

def download_orbits_with_reframe(
    project_root: Path,
    orbit: str,
    orbit_list: Path,
    pins: tuple,
    parallel_config: dict = None
) -> tuple[bool, list, str]:
    """..."""

    # ... existing mode 1 code (orbit download) ...

    # Check if parallel reframing is enabled
    parallel_config = parallel_config or {}
    use_parallel = parallel_config.get("parallel_reframe", False)
    reframe_workers = parallel_config.get("reframe_workers", 4)

    if use_parallel:
        # Parallel reframing
        print("\nUsing parallel reframing...")
        success, results = run_parallel_reframe(
            project_root,
            orbit,
            num_workers=reframe_workers
        )
        # Convert results to standard format
        commands.append({
            "command": "parallel_reframe",
            "success": success,
            "dates_processed": len(results),
            "dates_failed": sum(1 for r in results if not r['success'])
        })
    else:
        # Original sequential reframing (mode 2)
        cmd2 = f"organize_files_tops_linux.csh {orbit_list} {pins_file} 2"
        rc2, stdout2, stderr2 = run_cmd(cmd2)
        # ... existing code ...

    return success, commands, result_msg
```

### שינויים ב-config

```yaml
# config_track58.yaml
parallel:
  num_cores: 12
  parallel_subswaths: true
  # Parallel reframing (Stage 4)
  parallel_reframe: true
  reframe_workers: 4  # Number of parallel reframe workers
```

### שינויים ב-main.py

```python
# ב-stage_04_download_orbits:
parallel_config = self.config.get("parallel", {})

success, commands, msg = run_download_orbits(
    project_root=self.project_root,
    orbit=self.orbit,
    # ... existing params ...
    parallel_config=parallel_config
)
```

---

## 10. הערכת ביצועים

### מדידות בסיסיות

| מטריקה | ערך נוכחי | ערך צפוי (4 workers) |
|--------|-----------|---------------------|
| זמן לתאריך בודד | ~20 שניות | ~20 שניות (לא משתנה) |
| סה"כ 122 תאריכים (סדרתי) | ~40 דקות | - |
| סה"כ 122 תאריכים (מקבילי) | - | ~10-12 דקות |
| שיפור צפוי | - | ~3.5x |

### הערות

1. **Overhead**: יצירת תיקיות וsymlinks מוסיפה ~1-2 שניות לכל תאריך
2. **I/O bound**: הריפריימינג הוא בעיקר קריאה/כתיבה לדיסק, אז יותר מ-4-6 workers לא בהכרח יעזור
3. **Memory**: כל worker צורך ~500MB RAM, אז 4 workers = ~2GB
4. **CPU**: הסקריפט משתמש ב-100% core אחד, אז עם 4 workers נצפה ל-~400% CPU

### המלצה לכמות workers

| משאבים זמינים | workers מומלץ |
|---------------|---------------|
| 4 cores, 8GB RAM | 2-3 |
| 8 cores, 16GB RAM | 4-6 |
| 16 cores, 32GB RAM | 6-8 |
| 32+ cores, 64GB+ RAM | 8-12 |

---

## נספח: רשימת קבצים לשינוי

| קובץ | סוג שינוי | תיאור |
|------|----------|-------|
| `gmtsar_wrapper/parallel_reframe.py` | **חדש** | מודול המקבול המלא |
| `gmtsar_wrapper/make_orbits_04.py` | עריכה | אינטגרציה עם המודול החדש |
| `gmtsar_wrapper/main.py` | עריכה | העברת parallel_config |
| `config_example.yaml` | עריכה | הוספת אופציות חדשות |
| `docs/PARALLEL_REFRAME_PLAN.md` | **חדש** | תיעוד זה |

---

## היסטוריית גרסאות

| תאריך | גרסה | שינויים |
|-------|------|---------|
| 2026-02-04 | 1.0 | יצירה ראשונית |
