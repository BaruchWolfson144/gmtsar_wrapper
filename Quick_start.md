# מדריך התחלה מהירה - GMTSAR Python Wrapper

## סקירה כללית

הפרויקט מאפשר הרצת תהליך SBAS InSAR מלא באמצעות ממשק Python מודולרי. התוכנה כוללת 8 שלבים מעיבוד נתוני Sentinel-1 ועד לפתיחת הפאזה.

## מה השתנה?

### קבצים חדשים שנוצרו:
1. **main.py** - נקודת הכניסה המרכזית לכל הפרוייקט
2. **config_example.yaml** - קובץ תצורה לדוגמה בפורמט YAML
3. **config_example.json** - קובץ תצורה לדוגמה בפורמט JSON
4. **README_USAGE.md** - מדריך משתמש מפורט באנגלית
5. **test_integration.py** - בדיקות אינטגרציה
6. **IMPLEMENTATION_SUMMARY.md** - תיעוד טכני מפורט של המימוש

### קבצים ששונו:
- **run_interferograms_06.py** - תוקן באג + matplotlib הפך לאופציונלי

### כל שאר הקבצים נשארו ללא שינוי!
הקבצים make_dir_tree_01.py, stage_02_download_data.py, make_dem_03.py, make_orbits_04.py וכו' עדיין עובדים בדיוק כמו קודם.

## איך זה עובד?

### ניהול מצב (State Management)
הפיתרון המרכזי לבעיה שהעלית: איך מעבירים פרמטרים שמתגלים רק בזמן ריצה?

**התשובה**: המערכת שומרת מצב בקובץ JSON:
```
<project_root>/wrapper_meta/state/project_state.json
```

**דוגמה לזרימה**:
1. שלב 5 - בוחר אוטומטית את תמונת ה-master
2. שלב 5 - שומר את ה-master בקובץ המצב
3. שלב 6 - קורא את ה-master מהקובץ
4. שלב 7 - קורא את ה-master מהקובץ

**לא צריך לזכור פרמטרים בין שלבים!**

## שלושה אופני שימוש

### 1. ריצה רציפה אוטומטית (מומלץ)

```bash
# צור קובץ תצורה
cp config_example.yaml hawaii_project.yaml
# ערוך את הקובץ עם הפרמטרים שלך

# הרץ את כל השלבים
python main.py hawaii_project.yaml --sequential
```

המערכת תרוץ את כל 8 השלבים ברצף, תדלג על שלבים שכבר הושלמו, ותשמור מצב אחרי כל שלב.

### 2. ריצה שלב אחרי שלב

```bash
# שלב 1
python main.py /path/to/project --stage 1 --orbit asc

# שלב 2
python main.py /path/to/project --stage 2 \
  --minlon -157 --maxlon -154.2 --minlat 18 --maxlat 20.4

# שלב 4
python main.py /path/to/project --stage 4 --orbit-mode 1

# וכן הלאה...
```

### 3. המשך אחרי הפסקה

```bash
# התחלת עיבוד (רץ שלבים 1-5, ואז הפסקה)
python main.py config.yaml --sequential
# ^C (הפסקה)

# המשך מהשלב הבא
python main.py /path/to/project --resume
```

## בדיקת סטטוס

```bash
python main.py /path/to/project --status
```

יציג:
- אילו שלבים הושלמו
- פרמטרים שנשמרו (כולל ה-master שנבחר)
- מתי בוצע העדכון האחרון

## מבנה קובץ התצורה

**⚠️ חשוב - פורמט נתיבים**: בקבצי YAML, השתמש תמיד בסלשים קדמיים (/) ולא בסלשים הפוכים (\).
- ✓ נכון: `"/home/user/project"`
- ✗ לא נכון: `"C:\Users\project"` (יגרום לשגיאת YAML)

```yaml
project_root: "/path/to/project"
orbit: "asc"

dem:
  minlon: -157.0
  maxlon: -154.2
  minlat: 18.0
  maxlat: 20.4
  mode: 1

interferograms:
  threshold_time: 100
  threshold_baseline: 150

# ... עוד פרמטרים
```

ראה `config_example.yaml` לדוגמה מלאה עם הסברים.

## 8 השלבים

1. **stage_01** - יצירת מבנה תיקיות
2. **stage_02** - הורדת נתונים מ-ASF (Sentinel-1 data)
3. **stage_03** - הורדת DEM
4. **stage_04** - הורדת קבצי orbit + reframing (אופציונלי)
5. **stage_05** - בחירת master ו-alignment
6. **stage_06** - יצירת אינטרפרוגרמות
7. **stage_07** - מיזוג אינטרפרוגרמות
8. **stage_08** - פתיחת פאזה (unwrapping)

## בדיקה שהכל עובד

```bash
python3 test_integration.py
```

אם כל הבדיקות עוברות (✓), המערכת מוכנה לשימוש!

## דוגמאות שימוש

### דוגמה 1: פרויקט הוואי
```bash
# צור תיקייה חדשה
mkdir -p ~/insar_projects/hawaii

# צור קובץ תצורה
cat > hawaii_config.yaml << EOF
project_root: "~/insar_projects/hawaii"
orbit: "asc"

dem:
  minlon: -157.0
  maxlon: -154.2
  minlat: 18.0
  maxlat: 20.4
  mode: 1

orbits:
  mode: 1
  reframe: true
  pin1_lon: -157.0
  pin1_lat: 18.0
  pin2_lon: -154.2
  pin2_lat: 20.4

interferograms:
  threshold_time: 100
  threshold_baseline: 150

unwrap:
  coherence_threshold: 0.075
EOF

# הרץ את כל התהליך
python main.py hawaii_config.yaml --sequential
```

### דוגמה 2: הרצה מחדש של שלב ספציפי
```bash
# נניח שרוצים לשנות את threshold_time בשלב 6
python main.py ~/insar_projects/hawaii --stage 6 \
  --threshold-time 120 --threshold-baseline 200
```

### דוגמה 3: הרצה ידנית שלב אחרי שלב
```bash
PROJECT=~/insar_projects/hawaii

# שלב 1
python main.py $PROJECT --stage 1 --orbit asc

# בדיקת סטטוס
python main.py $PROJECT --status

# שלב 2
python main.py $PROJECT --stage 2 \
  --minlon -157 --maxlon -154.2 --minlat 18 --maxlat 20.4

# המשך...
```

## פתרון בעיות

### בעיה: "Master image not found"
**פתרון**: יש להריץ קודם את שלב 5:
```bash
python main.py /path/to/project --stage 5
```

### בעיה: קובץ המצב התקלקל
**פתרון**: מחק ותתחיל מחדש:
```bash
rm <project_root>/wrapper_meta/state/project_state.json
```

### בעיה: שלב נתקע
**פתרון**: בדוק את קבצי הלוג:
```bash
cat <project_root>/wrapper_meta/logs/step*.json
```

## תאימות לאחור

✓ כל הסקריפטים הישנים עדיין עובדים!
✓ אפשר להמשיך להשתמש בהם ישירות:
```bash
python make_dir_tree_01.py /path --orbit asc
python stage_02_download_data.py /path --polygon ... --start ... --end ...
python make_dem_03.py /path --minlon ... --maxlon ...
```

## דרישות מערכת

### חובה:
- Python 3.8 ומעלה
- GMTSAR 6.0 ומעלה
- Linux (נדרש עבור GMTSAR)

### אופציונלי:
- PyYAML (עבור קבצי YAML)
- matplotlib (עבור גרפים)

להתקנת PyYAML:
```bash
pip install pyyaml
```

## קבצים חשובים

- **main.py** - הסקריפט המרכזי
- **config_example.yaml** - תבנית לקובץ תצורה
- **README_USAGE.md** - מדריך מפורט באנגלית
- **IMPLEMENTATION_SUMMARY.md** - תיעוד טכני מפורט
- **test_integration.py** - בדיקות

## למידע נוסף

ראה את הקובץ `README_USAGE.md` למדריך מפורט באנגלית עם הסברים על כל פרמטר ושלב.

## סיכום

המערכת מספקת שלושה רבדים של שליטה:

1. **אוטומטי מלא**: `python main.py config.yaml --sequential`
2. **שליטה ידנית**: `python main.py project --stage N`
3. **המשך חכם**: `python main.py project --resume`

כל הפרמטרים נשמרים אוטומטית ומועברים בין השלבים.
אין צורך לזכור או להעביר ידנית את תמונת ה-master!

---

**בהצלחה!** 🚀
