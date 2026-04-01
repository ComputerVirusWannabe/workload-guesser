# Workload Guesser

A machine-learning system that predicts the **difficulty and workload level**
of college courses, helping students build balanced schedules before registration.

Workload Guesser combines **NLP-based text analysis** with **structured course
metadata** to classify every course as one of three workload tiers:

| Label  | Meaning |
|--------|---------|
| **low**    | Light workload — few assignments, mostly exams or attendance-based |
| **medium** | Moderate workload — regular homework and a couple of exams |
| **high**   | Heavy workload — weekly assignments, multiple exams, and/or projects |

## Features


| **TF-IDF text** | Uni- and bi-gram frequencies over the free-text course description (up to 500 features) |
| **Model training** | A random Forest classifier |
| **Keyword counts** | Counts of workload-related words: *exam*, *midterm*, *assignment*, *project*, *weekly*, *rigorous*, … |
| **Course metadata** | Normalised course level, credit hours, and average historical GPA |


## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Python 3.9 or later is required.

## Usage

### Interactive mode

```bash
python -m workload_guesser.cli
# or, after `pip install -e .`:
workload-guesser
```

The program trains on the built-in sample data, then prompts for course
details and gives a prediction with a confidence breakdown.

### Command-line prediction

```bash
workload-guesser predict \
  --department CS \
  --level 4000 \
  --credits 3 \
  --description "Weekly problem sets, two midterms, a final exam, and a semester project." \
  --gpa-avg 2.7
```

Example output:

```
Predicted workload:  HIGH

Confidence breakdown:
  low      3.0%
  medium  12.5%
  high   84.5%
```

### Train and save a model

```bash
workload-guesser train --save models/workload.pkl
```

### Load a saved model for prediction

```bash
workload-guesser predict \
  --model models/workload.pkl \
  --department MATH --level 4000 --credits 3 \
  --description "Rigorous proof-based course with weekly homework and three exams."
```

### Python API

```python
from workload_guesser import WorkloadPredictor
from workload_guesser.data import course_to_dataframe

# Train
predictor = WorkloadPredictor()
predictor.train()          # uses built-in data by default

# Predict a single course
df = course_to_dataframe(
    department="CS",
    level=4000,
    credits=3,
    description="Rigorous algorithms course: weekly problem sets, two midterms, final.",
    gpa_avg=2.6,
)
label = predictor.predict(df)[0]        # 'low' | 'medium' | 'high'
proba = predictor.predict_proba(df)     # {'low': 0.03, 'medium': 0.15, 'high': 0.82}

# Save and reload
predictor.save("models/workload.pkl")
loaded = WorkloadPredictor.load("models/workload.pkl")
```

## Running tests

```bash
pytest
```

## Training on own data

Prepare a CSV with at least these columns:

| Column        | Type   | Description |
|---------------|--------|-------------|
| `department`  | str    | Dept. code (e.g. `CS`) |
| `level`       | int    | Course level (e.g. `3000`) |
| `credits`     | int    | Credit hours |
| `description` | str    | Free-text course description |
| `workload`    | str    | `low`, `medium`, or `high` |

Optional columns (`gpa_avg`, `num_assignments`, `num_exams`, `num_projects`)
improve predictions when available.

```bash
workload-guesser train --data my_courses.csv --save models/custom.pkl
```


