# MODEL DOCTOR — AGENT RULES

## 1. Read Before Coding

Before doing any work:

1. Read `README.md` completely.
2. Read `AGENTS.md` completely.
3. Read `PROGRESS.md`.
4. Inspect the current project structure.
5. Run `git status`.
6. Identify the current incomplete level.

Do not start coding before doing these steps.

---

# 2. Main Goal

Build Model Doctor.

Model Doctor diagnoses the fault type of small MNIST CNN classifiers using measurable diagnostic behaviour.

The four diagnosis classes are:

```text
healthy
class_imbalance
label_noise
shortcut
```

Data drift and multiple simultaneous faults are out of scope.

---

# 3. Core Architecture

The project uses:

```text
MNIST
    ↓
fault injection
    ↓
100 TinyCNN models
    ↓
diagnostic feature extraction
    ↓
Random Forest Model Doctor
    ↓
diagnosis
    ↓
Grad-CAM
    ↓
React frontend
```

The final frontend architecture is:

```text
React
   ↓
frontend/src/api/client.js
   ↓
static JSON
or
FastAPI
or
Spring Boot → FastAPI
```

---

# 4. Model Zoo Rules

The target is:

```text
100 models total

25 healthy
25 class_imbalance
25 label_noise
25 shortcut
```

Use one TinyCNN architecture.

Use:

```text
20,000 training images
10,000 clean test images
5 epochs
batch size 256
Adam
learning rate 1e-3
```

Use random seeds and severity values generated from a reproducible master seed.

Save metadata:

```text
model_id
fault
severity
seed
```

Fault injection must be deterministic for the same seed.

---

# 5. Fault Injection Rules

## Class imbalance

Keep only a small fraction of training images for 1-3 selected digits.

Severity maps to approximately:

```text
20% → 2%
```

of the affected class data.

The selected digits must be deterministic for a given seed.

---

## Label noise

Randomly change 10%-50% of training labels.

The replacement label must be different from the original label.

The operation must be deterministic for the same seed.

---

## Shortcut

Add a small corner patch to training images.

The patch mapping depends on the class.

The training labels remain unchanged.

Test images must remain clean.

The same mapping must be used by:

```text
fault injection
probe generation
diagnostic features
```

---

## Healthy

Do not modify the training data.

---

# 6. Diagnostic Features

The required features are exactly:

```text
acc_clean
acc_min_class
acc_std_class
mean_confidence
mean_entropy
conf_wrong_rate
train_val_loss_gap
train_conf_minus_val_conf
acc_patch_matched
acc_patch_mismatched
```

The patch probes must be applied to all models.

---

# 7. Strict Leakage Rules

Never use these as Model Doctor input features:

```text
model_id
fault
severity
```

`fault` is the target.

`model_id` is only an identifier.

`severity` is metadata.

Do not create artificial features that directly reveal the fault.

If a result is suspiciously perfect, investigate leakage.

---

# 8. Model Doctor

Use:

```text
RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
```

Use:

```text
random_state=42
```

for the main split.

Use an 80/20 stratified train/test split.

Assert that train and test model IDs are disjoint.

Also perform 5-fold stratified cross-validation.

Report:

```text
accuracy
precision
recall
F1
confusion matrix
feature importance
CV mean
CV standard deviation
```

Compare with:

```text
Logistic Regression
majority-class dummy baseline
```

---

# 9. Diagnosis Rule

The diagnosis MUST come from the trained Random Forest.

The LLM must never decide:

```text
healthy
class_imbalance
label_noise
shortcut
```

If an LLM is used, it may only explain an already-generated diagnosis.

---

# 10. Diagnose Function

The central function is:

```python
diagnose(model_id: str) -> dict
```

It must remain independent from the UI.

`src/diagnose.py` must not import:

```text
streamlit
fastapi
flask
```

or another web framework.

---

# 11. API Contract

The JSON structure is fixed by:

```text
docs/API_CONTRACT.md
```

Do not change the contract casually.

If a contract change is genuinely necessary:

1. update the contract document
2. update the implementation
3. update tests
4. document the reason in `PROGRESS.md`

---

# 12. React Rules

The React application uses:

```text
React
Vite
JavaScript
Tailwind CSS
Recharts
```

Only this file may fetch data:

```text
frontend/src/api/client.js
```

Components must never directly use:

```javascript
fetch(...)
```

The API client must support:

```text
VITE_API_MODE=static
VITE_API_MODE=api
```

Static mode uses:

```text
frontend/public/data/
```

API mode uses the backend.

Both modes must return the same JSON shape.

---

# 13. Grad-CAM

Use the final convolutional layer of TinyCNN.

Do not use an external Grad-CAM library.

The heatmap must be:

```text
28 x 28
```

and normalized to:

```text
0-1
```

Test for:

* correct shape
* valid values
* no NaN

---

# 14. Testing Rules

Every major module must have tests.

At minimum test:

```text
data
fault injection
probes
training
model zoo
features
doctor
Grad-CAM
diagnose
backend
```

Run:

```bash
pytest -q
```

before marking a level complete.

Do not ignore failing tests.

Do not delete tests just because they fail.

Fix the implementation or document a genuine limitation.

---

# 15. Development Levels

Work in this exact order:

```text
Level 0: Scaffold

Level 1: Data, CNN, fault injection, probes

Level 2: Model zoo

Level 3: Feature extraction

Level 4: Meta-classifier and evaluation

Level 5A: Diagnose function, Grad-CAM, data export

Level 5B: React frontend

Level 6: Report material and viva questions

Level 7: Extra comparison and ablation

Level 8: FastAPI backend

Level 9: Spring Boot layer
```

Level 7 is optional.

Level 8 is optional.

Level 9 is optional.

Do not start optional levels before the required project works.

---

# 16. Level Discipline

Work ONLY on the current level.

Do not jump ahead.

Do not build future functionality early unless the current level explicitly requires it.

Do not rewrite completed levels without a strong reason.

Do not delete completed work.

If a refactor is necessary:

1. preserve behaviour
2. run tests
3. document the refactor in `PROGRESS.md`

---

# 17. Model Zoo Safety

Before running all 100 models, run:

```bash
python -m src.train_zoo --limit 4
```

Check that it works.

The zoo must be resumable.

Existing model files must be skipped.

A failure for one model must be logged and should not unnecessarily stop the whole zoo.

Do not commit large `.pt` model files to git.

---

# 18. Reproducibility

Use fixed seeds wherever reproducibility is required.

Document:

* subset seed
* master seed
* model seeds
* severity ranges
* patch mapping
* epochs
* batch size
* learning rate

The same metadata must allow the faulty training data to be rebuilt.

---

# 19. Results

Never invent results.

Never write fake:

```text
accuracy
precision
recall
F1
CV score
feature importance
timing
```

If a result has not been measured, say:

```text
not measured yet
```

If an experiment fails, report the failure.

---

# 20. PROGRESS.md

After every level:

1. update `PROGRESS.md`
2. record what was created
3. record tests
4. record actual results
5. record important decisions
6. record known issues
7. record next step

Never mark a level complete before its required tests and validation have actually passed.

---

# 21. Git

Commit after every completed level.

Commit messages:

```text
Level 0: scaffold
Level 1: data, CNN, fault injection, probes
Level 2: model zoo
Level 3: feature extraction
Level 4: doctor and evaluation
Level 5A: diagnose function, gradcam, export
Level 5B: React frontend
Level 6: docs and viva material
Level 7: ablation and comparison
Level 8: FastAPI backend
Level 9: Spring Boot layer
```

Before and after important work:

```bash
git status
git log --oneline -10
```

---

# 22. Code Quality

Prefer simple, readable code.

Use type hints and docstrings for core Python functions.

Avoid unnecessary abstractions.

Do not introduce frameworks that are not required.

Do not add dependencies without a reason.

Keep the ML logic independent from the UI.

Keep backend logic independent from React.

---

# 23. Scientific Integrity

This is a research prototype.

Do not optimize the system only to make the numbers look good.

If the doctor performs poorly:

* inspect the confusion matrix
* inspect feature distributions
* check leakage
* check fault injection
* check severity
* document the limitation

Do not hide poor results.

---

# 24. Time Priority

If compute or time becomes limited:

1. correct model zoo
2. correct feature extraction
3. correct evaluation
4. working diagnosis
5. React demo
6. Grad-CAM
7. UI polish
8. optional APIs

Evaluation must not be removed.

Grad-CAM may be reduced or skipped if necessary.

---

# 25. Communication With the User

At the end of every level, report:

```text
1. What was created
2. What was changed
3. What was tested
4. Test result
5. Actual experiment result
6. Git commit
7. Current level
8. Next level
9. Known issues
```

Keep the report factual.

Do not claim success without evidence.

---

# 26. Stop Conditions

Stop and ask for direction if:

* a requirement conflicts with the README
* a destructive change may be required
* existing completed work may be lost
* an experiment produces unexpected results that require a design change
* a major dependency or architecture change is required
* the JSON API contract must be changed
* a scientific assumption needs to be changed

Do not silently make major design decisions.

---

# 27. Final Rule

Build the project incrementally.

Read the project files before coding.

Work only on the current level.

Test everything.

Record actual results.

Commit every completed level.

Never invent results.

Never use an LLM as the source of the diagnosis.

Never sacrifice scientific correctness for appearance.
