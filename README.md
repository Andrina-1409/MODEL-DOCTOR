# Model Doctor

## 1. Project Summary

Model Doctor is an AI system that diagnoses why a small machine learning image classifier is failing.

The system creates a controlled model zoo of small CNN classifiers trained on MNIST. Some models are healthy, while others contain one deliberately injected fault:

* class imbalance
* label noise
* shortcut learning
* healthy model

Diagnostic signals are extracted from each trained model. A Random Forest meta-classifier, called the **Model Doctor**, learns to identify the fault from those signals.

The final system provides:

* fault diagnosis
* diagnosis confidence
* probabilities for all four fault classes
* diagnostic feature values
* Grad-CAM visual evidence
* suggested fix
* React frontend
* static demo mode
* optional FastAPI backend
* optional Spring Boot layer

The diagnosis itself must always come from the trained meta-classifier. An LLM must never decide the diagnosis.

---

# 2. Problem Statement

When a machine learning model performs badly, developers often debug it manually by checking data, predictions, confidence, class performance and visual behaviour.

This can be slow and requires expertise.

Model Doctor attempts to automate the first stage of this debugging process by measuring observable model behaviour and predicting which known fault is affecting the model.

This project is a controlled research prototype, not a general-purpose model debugging system.

---

# 3. Objectives

The project must:

1. Build a labeled model zoo of approximately 100 small CNNs.
2. Use MNIST as the dataset.
3. Create 25 models for each fault category.
4. Inject faults in a deterministic and reproducible way.
5. Extract measurable diagnostic features from every model.
6. Train a Random Forest meta-classifier.
7. Evaluate the doctor using a proper train/test split and cross-validation.
8. Prevent model/data leakage.
9. Provide a `diagnose(model_id)` Python function.
10. Generate Grad-CAM visual evidence.
11. Export static JSON data for a React frontend.
12. Build a React + Vite frontend.
13. Keep the system backend-ready.
14. Prepare report, PPT and viva material from actual project results.
15. Keep the implementation reproducible and testable.

---

# 4. Project Scope

## Included

* MNIST
* one TinyCNN architecture
* 100 models
* four classes:

  * healthy
  * class_imbalance
  * label_noise
  * shortcut
* synthetic single faults
* diagnostic feature extraction
* Random Forest meta-classifier
* evaluation
* Grad-CAM
* React frontend
* static JSON demo
* optional FastAPI backend
* optional Spring Boot layer

## Out of Scope

The current version does not attempt to support:

* data drift
* multiple simultaneous faults
* multiple CNN architectures
* large production models
* CIFAR-10
* text models
* tabular models
* automatic model repair

These can be discussed as future work.

---

# 5. Fault Definitions

| Fault             | Injection                                                             | Expected Behaviour                                                               |
| ----------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `class_imbalance` | Keep only a small fraction of training images for 1-3 selected digits | Very low recall for affected digits and large variation in per-class accuracy    |
| `label_noise`     | Randomly change 10%-50% of training labels to another class           | Lower confidence, weaker generalization and different train/validation behaviour |
| `shortcut`        | Add a small class-dependent corner patch to training images           | High normal accuracy but behaviour changes strongly when the patch is mismatched |
| `healthy`         | No fault injection                                                    | Balanced and confident behaviour                                                 |

The shortcut patch must only be injected into training images.

Clean test images must remain clean.

The patch mapping must be fixed and documented.

---

# 6. Model Zoo

The target model zoo is:

* 100 total models
* 25 healthy
* 25 class imbalance
* 25 label noise
* 25 shortcut

Every model must have metadata:

```text
model_id
fault
severity
seed
```

The model zoo must be reproducible.

Fault injection must be deterministic when the same seed is used.

The model zoo must be resumable.

If a model file already exists, training should skip that model.

Failures during zoo generation should be logged without stopping the entire zoo.

---

# 7. Dataset

Use MNIST through torchvision.

Use:

* 20,000 training images
* 10,000 clean test images
* images loaded into memory
* fixed subset selection using a seed
* pixel values converted to float32
* pixel values normalized to [0,1]

Training configuration:

```text
epochs = 5
batch_size = 256
optimizer = Adam
learning_rate = 1e-3
```

Use GPU when available.

Do not reload MNIST separately for every model.

---

# 8. TinyCNN Architecture

Use one small CNN architecture for all models.

Architecture:

```text
Conv2d(1, 8, 3)
ReLU
MaxPool

Conv2d(8, 16, 3)
ReLU
MaxPool

Flatten
Linear(..., 10)
```

The architecture must expose:

* final convolutional feature maps
* penultimate features

The final convolutional layer is required later for Grad-CAM.

Do not introduce multiple architectures in this version.

---

# 9. Diagnostic Features

Every model must produce exactly one feature row.

Required features:

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

## `acc_clean`

Accuracy on the clean MNIST test set.

## `acc_min_class`

Worst accuracy among the ten digit classes.

Useful for identifying class imbalance.

## `acc_std_class`

Standard deviation of the ten per-class accuracies.

Useful for identifying uneven class performance.

## `mean_confidence`

Mean maximum softmax probability on the clean test set.

## `mean_entropy`

Mean prediction entropy on the clean test set.

## `conf_wrong_rate`

Fraction of all test samples where:

```text
prediction is wrong
AND
confidence > 0.9
```

## `train_val_loss_gap`

Loss on a fixed sample of the model's own faulty training data minus loss on clean validation data.

## `train_conf_minus_val_conf`

Mean training confidence minus mean validation confidence.

## `acc_patch_matched`

Accuracy when the correct class-dependent patch is added to test images.

## `acc_patch_mismatched`

Accuracy when a wrong class-dependent patch is added.

A large difference between matched and mismatched behaviour can reveal shortcut learning.

The same patch probes must be applied to every model, including healthy models.

The doctor must not know the ground-truth fault while extracting features.

---

# 10. Feature Leakage Rules

The following must NEVER be used as doctor input features:

```text
model_id
fault
severity
```

The Random Forest must only receive measurable diagnostic behaviour.

`fault` is the target label.

`severity` is metadata and must not be used as a feature.

`model_id` must not be used as a feature.

Never create a feature that directly reveals the injected fault.

---

# 11. Model Doctor

The Model Doctor is a Random Forest classifier.

Use:

```text
RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
```

The target is:

```text
fault
```

Classes:

```text
healthy
class_imbalance
label_noise
shortcut
```

The doctor must be trained only on the training portion of the model dataset.

---

# 12. Evaluation

Use an 80/20 stratified train/test split.

Use:

```text
random_state = 42
```

Each row represents exactly one model.

Never generate multiple independent feature rows from the same model and split them between train and test.

Assert that train and test model IDs are disjoint.

Report:

* accuracy
* precision
* recall
* F1
* confusion matrix
* feature importance
* 5-fold cross-validation mean accuracy
* 5-fold cross-validation standard deviation

Also compare against:

* Logistic Regression
* majority-class dummy baseline

The evaluation is one of the most important parts of the project.

Do not remove or weaken evaluation to improve the appearance of the demo.

---

# 13. Scientific Integrity

Never invent:

* accuracy
* precision
* recall
* F1
* cross-validation results
* feature importance
* confusion matrix results
* timings
* model counts
* diagnosis results

All reported numbers must come from actual execution.

If something fails, report the failure.

If results are unexpectedly poor, investigate instead of hiding the result.

If accuracy is suspiciously perfect, investigate possible leakage.

---

# 14. Grad-CAM

Implement Grad-CAM manually using PyTorch hooks.

It must operate on the final convolutional layer.

Required functions:

```text
gradcam(...)
overlay(...)
corner_attention_score(...)
```

Grad-CAM output:

```text
28 x 28
```

Values:

```text
0 to 1
```

The system should also calculate attention mass in the four 6x6 image corners.

---

# 15. Diagnose Function

Create:

```text
src/diagnose.py
```

with:

```python
diagnose(model_id: str) -> dict
```

This is the central diagnosis function.

It must:

1. Load the model.
2. Rebuild the model's training data deterministically.
3. Extract diagnostic features.
4. Load the trained Random Forest.
5. Generate diagnosis probabilities.
6. Select the diagnosis from the Random Forest output.
7. Generate Grad-CAM information.
8. Return the required JSON-compatible structure.

The module must not import:

```text
streamlit
fastapi
flask
```

or other UI/web libraries.

The diagnosis must come from the trained Model Doctor.

An LLM may only be used later to convert the diagnosis into natural-language explanation.

---

# 16. Diagnosis JSON Contract

The diagnosis response must contain:

```json
{
  "model_id": "model_0001",
  "diagnosis": "shortcut",
  "confidence": 0.92,
  "probabilities": {
    "healthy": 0.02,
    "class_imbalance": 0.01,
    "label_noise": 0.05,
    "shortcut": 0.92
  },
  "features": {},
  "gradcam_images": [],
  "corner_attention": 0.0,
  "suggested_fix": "remove or randomize the spurious cue and use augmentation",
  "ground_truth": "shortcut"
}
```

The exact contract must be documented in:

```text
docs/API_CONTRACT.md
```

Do not change the JSON contract without updating that document.

---

# 17. Suggested Fixes

Use a fixed mapping.

```text
healthy
→ no fault detected

class_imbalance
→ rebalance the data or use a weighted loss

label_noise
→ clean the labels or use a robust loss

shortcut
→ remove or randomize the spurious cue and use augmentation
```

The suggested fix is an explanation/recommendation.

It is not used to determine the diagnosis.

---

# 18. React Architecture

Use:

```text
React
Vite
JavaScript
Tailwind CSS
Recharts
```

The frontend must communicate through:

```text
frontend/src/api/client.js
```

This must be the ONLY frontend file that performs data fetching.

Components must never directly call `fetch()`.

Support:

```text
VITE_API_MODE=static
```

and:

```text
VITE_API_MODE=api
```

Static mode reads:

```text
frontend/public/data/
```

API mode communicates with the backend.

The React component code should remain unchanged when switching between static and API modes.

---

# 19. React Components

The frontend should contain:

### Header

Explain Model Doctor briefly.

### ModelSelector

Allow selection of the example models.

### DiagnosisCard

Display:

```text
Diagnosis: <fault>
Confidence: <percentage>
```

### ProbabilityBars

Show probabilities for:

* healthy
* class imbalance
* label noise
* shortcut

### FeatureTable

Show each diagnostic feature and its human-readable meaning.

### GradcamGallery

Show the Grad-CAM images.

Show corner attention.

For shortcut diagnosis, show an explanation that the model may be relying on the corner patch.

### SuggestedFix

Show the suggested fix.

### GroundTruthBadge

Clearly label ground truth as:

```text
for demo only
```

Show whether the doctor's prediction matches the known injected fault.

### Loading/Error States

Every data-dependent component must handle loading and errors.

---

# 20. Static Demo

The first frontend version must work without a backend.

Export approximately eight example models:

```text
2 healthy
2 class_imbalance
2 label_noise
2 shortcut
```

Choose the examples from the test split used by Level 4.

Export:

```text
frontend/public/data/models.json
frontend/public/data/examples.json
frontend/public/data/diagnosis/<id>.json
frontend/public/data/gradcam/<id>_<k>.png
```

The React app must work using only these static files.

---

# 21. Project Structure

The final core structure should be:

```text
model_doctor/
├── AGENTS.md
├── PROGRESS.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── fault_injection.py
│   ├── probes.py
│   ├── train_zoo.py
│   ├── extract_features.py
│   ├── train_doctor.py
│   ├── gradcam.py
│   ├── diagnose.py
│   └── export_demo_data.py
│
├── frontend/
│   ├── public/
│   │   └── data/
│   └── src/
│       ├── api/
│       │   └── client.js
│       └── components/
│
├── backend/
│
├── models/
├── features/
├── reports/
├── docs/
├── tests/
└── notebooks/
```

---

# 22. Required Tests

Use pytest.

Every major module must have tests.

Required test areas:

```text
data
fault injection
probes
training
model zoo
feature extraction
doctor
Grad-CAM
diagnosis
frontend build
backend
```

Tests must verify actual behaviour, not just that functions import successfully.

---

# 23. Development Levels

Build the project in this exact order.

```text
Level 0
Scaffold

Level 1
Data, CNN, fault injection, probes

Level 2
Model zoo

Level 3
Feature extraction

Level 4
Meta-classifier and evaluation

Level 5A
Diagnose function, Grad-CAM, data export

Level 5B
React frontend

Level 6
Report material and viva questions

Level 7
Extra comparison and ablation

Level 8
FastAPI backend

Level 9
Spring Boot layer
```

Do not jump ahead.

Level 7 is optional.

Level 8 is optional and should be done after the main submission.

Level 9 is optional and should be done after the main submission.

---

# 24. Level 0

Create the project scaffold.

Create:

```text
src/
frontend/
models/
features/
reports/
docs/
tests/
notebooks/
```

Add `.gitkeep` where required.

Create:

```text
requirements.txt
.gitignore
```

Requirements:

```text
torch
torchvision
scikit-learn
pandas
numpy
matplotlib
pytest
joblib
```

Do not add FastAPI yet.

Do not add Spring Boot yet.

Do not write ML implementation code during Level 0.

Initialize git if required.

Create the first commit:

```text
Level 0: scaffold
```

---

# 25. Level 1

Implement:

```text
src/data.py
src/model.py
src/train.py
src/fault_injection.py
src/probes.py
```

Add tests.

Verify:

* MNIST loading
* TinyCNN
* training
* imbalance injection
* label noise injection
* shortcut injection
* healthy model
* deterministic seeds
* patch probes

Train a small sanity example.

The shortcut model should show a clearly larger mismatched-patch accuracy drop than a healthy model.

If it does not, investigate and document the adjustment.

---

# 26. Level 2

Implement:

```text
src/train_zoo.py
```

Create the 100-model zoo.

Before the full run, test:

```bash
python -m src.train_zoo --limit 4
```

This is mandatory before starting the complete zoo.

The script must be resumable.

Only mark Level 2 complete when all 100 model files exist.

Do not commit large `.pt` model files to git.

Record where the model files are stored.

---

# 27. Level 3

Implement:

```text
src/extract_features.py
```

Generate:

```text
features/features.csv
```

It must contain:

```text
100 rows
```

plus metadata columns and all required diagnostic features.

Check:

* no NaN
* no infinity
* valid accuracy values
* valid confidence values
* correct columns
* one row per model

Create:

```text
reports/feature_summary.csv
```

---

# 28. Level 4

Implement:

```text
src/train_doctor.py
```

Generate:

```text
doctor.pkl
reports/confusion_matrix.png
reports/feature_importance.png
reports/metrics.json
reports/classification_report.txt
```

Run:

* Random Forest
* Logistic Regression
* dummy baseline
* 5-fold cross-validation

Investigate suspiciously perfect accuracy.

Do not use:

```text
severity
model_id
fault
```

as input features.

---

# 29. Level 5A

Implement:

```text
src/gradcam.py
src/diagnose.py
src/export_demo_data.py
docs/API_CONTRACT.md
```

Create the static demo data.

Test:

```text
diagnose()
probabilities
diagnosis
Grad-CAM
JSON export
```

---

# 30. Level 5B

Build the React frontend.

The frontend must work in static mode.

Verify:

```bash
npm install
npm run build
```

Then run the development server and verify the complete demo.

---

# 31. Level 6

Create:

```text
docs/report_outline.md
docs/ppt_outline.md
docs/viva_questions.md
docs/demo_script.md
docs/explain_each_file.md
```

All numerical results must come from the actual project.

Never invent results.

---

# 32. Level 7 — Optional

Implement:

* feature ablation
* model comparison
* severity analysis

Do not overwrite existing reports.

---

# 33. Level 8 — Optional

Add FastAPI only after the core project and frontend are working.

Endpoints:

```text
GET /api/health
GET /api/models
GET /api/diagnose/{model_id}
```

The backend must call:

```text
src.diagnose.diagnose()
```

It must not duplicate diagnosis logic.

React must continue using the same API client.

---

# 34. Level 9 — Optional

Add a Spring Boot layer between React and FastAPI.

Use:

```text
Java 17
Spring Boot 3
Maven
```

Spring Boot should:

* proxy models
* proxy diagnosis
* store diagnosis history
* expose history
* proxy Grad-CAM images
* handle FastAPI failures
* keep the same JSON contract

Do not modify the Python diagnosis logic.

Do not modify the React components unnecessarily.

---

# 35. Git Rules

Create a git commit after every completed level.

Use these commit messages:

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

Before switching accounts or machines:

```bash
git status
git log --oneline -10
```

Commit or safely stash unfinished work.

---

# 36. Agent Workflow

Every coding session must follow this sequence:

1. Read `README.md`.
2. Read `AGENTS.md`.
3. Read `PROGRESS.md`.
4. Inspect the current project.
5. Check git status.
6. Identify the current level.
7. Work only on that level.
8. Run tests.
9. Fix failures.
10. Update `PROGRESS.md`.
11. Review the changes.
12. Commit the level.
13. Report exactly what was completed.

Never silently skip tests.

Never mark a level complete without actually completing and testing it.

Never start the next level automatically.

---

# 37. If Something Fails

Do not hide the failure.

Record:

```text
what failed
why it failed
what was changed
what test was run
whether the issue is resolved
```

If the problem cannot be resolved safely, stop and report it.

Do not invent a successful result.

---

# 38. Compute and Time Priority

The priority order is:

```text
1. Correct model zoo
2. Correct feature extraction
3. Correct evaluation
4. Correct diagnosis
5. React demo
6. Grad-CAM
7. UI polish
8. Optional backend layers
```

If time becomes limited:

1. Drop Grad-CAM first.
2. Then reduce UI polish.
3. Never drop evaluation.
4. Never invent results.

The notebook/reports can be used as a fallback demonstration.

---

# 39. Limitations

The final report and viva must clearly state:

* only MNIST is used
* only one TinyCNN architecture is used
* faults are synthetic
* only one fault is injected at a time
* only three fault types plus healthy are included
* the model zoo contains only 100 models
* the system may not transfer directly to larger models
* real-world models may contain multiple faults
* results from a small meta-dataset can have variance

---

# 40. Future Work

Possible future work:

* data drift diagnosis
* multiple simultaneous faults
* more CNN architectures
* CIFAR-10
* text models
* tabular models
* automatic repair suggestions
* testing public real-world models
* larger model zoos
* additional diagnostic signals

---

# 41. Final Validation

Before calling the project complete:

```bash
pytest -q
```

must pass.

Check:

```text
features/metadata.csv
features/features.csv
```

must contain the expected 100 model rows.

Check that reports contain:

```text
confusion matrix
feature importance
metrics
classification report
```

Check that:

```bash
npm install
npm run build
```

works.

Check that the React static demo works.

Check that the user can explain:

* all four fault classes
* every diagnostic feature
* why Random Forest is used
* why model-level splitting matters
* what leakage means
* what cross-validation means
* what Grad-CAM does
* limitations
* future work

---

# 42. Final Rule

The project must be built as a real, reproducible research prototype.

The coding agent is responsible for implementing and testing the software.

The developer is responsible for understanding the implementation and being able to explain it during the viva.

Never sacrifice scientific correctness just to make the demo look better.
