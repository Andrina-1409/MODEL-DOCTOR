# MODEL DOCTOR — PROJECT PROGRESS

> Last updated: 2026-10-02
>
> Current status: **Core ML pipeline complete. Frontend is running locally. End-to-end UI validation and final documentation/demo polish remain.**

---

## 1. PROJECT OVERVIEW

### Project name
**Model Doctor**

### Core idea
Model Doctor is an AI-assisted diagnostic system that tries to answer:

> **"Why did this machine-learning model fail?"**

Instead of only reporting model accuracy, the system performs controlled behavioural tests on a trained image classifier, extracts diagnostic signals, and uses a second machine-learning model ("the Doctor") to predict the likely failure mode.

The current controlled failure modes are:

1. Healthy
2. Label Noise
3. Class Imbalance
4. Shortcut Learning

The project uses MNIST-style image classification and deliberately injected faults so that the Doctor can be trained on models with known ground-truth failure modes.

---

# 2. PROJECT PIPELINE

The implemented pipeline is:

```text
MNIST data
    |
    v
Train controlled models
    |
    +--> Healthy
    +--> Label Noise
    +--> Class Imbalance
    +--> Shortcut
    |
    v
100-model zoo
    |
    v
Behavioural probes
    |
    +--> Clean accuracy
    +--> Faulty-data accuracy
    +--> Patch/mismatch probe
    +--> Corner attention
    +--> Additional diagnostic statistics
    |
    v
Feature extraction
    |
    v
Doctor classifier
    |
    v
Diagnosis
    |
    +--> Predicted failure mode
    +--> Probabilities
    +--> Diagnostic evidence
    +--> Suggested fix
    +--> Grad-CAM evidence
    |
    v
React frontend
```

---

# 3. LEVEL 0 — PROJECT SCAFFOLD

## Status: COMPLETE

The project structure was created and validated.

Important directories/files include:

```text
src/
tests/
features/
artifacts/
reports/
frontend/
PROGRESS.md
README.md
```

The required scaffold tests pass.

### Validation

```text
tests/test_scaffold.py
```

Result:

```text
2/2 scaffold tests passed
```

---

# 4. LEVEL 1 — DATA + MODEL + CONTROLLED FAULTS + PROBES

## Status: COMPLETE

The basic ML infrastructure is implemented and tested.

---

## 4.1 MNIST data handling

Implemented:

- MNIST-style dataset loading
- deterministic train/test selection
- reproducible random seeds
- normalized image values
- expected tensor shapes and dtypes

Tests cover:

- dataset shape
- dtype
- normalization
- reproducibility
- seed-dependent selection

Result:

```text
5/5 data tests passed
```

---

## 4.2 TinyCNN model

Implemented the project classifier:

```text
TinyCNN
```

The model exposes the necessary feature/penultimate representation used later for diagnosis and Grad-CAM.

Tests verify:

- output shape
- feature shape
- penultimate representation shape
- trainable parameters
- deterministic evaluation behaviour

Result:

```text
5/5 model tests passed
```

---

## 4.3 Controlled fault injection

Implemented four controlled conditions:

### Healthy

No intentional corruption.

### Label Noise

A controlled fraction of training labels is changed.

### Class Imbalance

Selected classes are deliberately reduced to create an imbalanced training distribution.

### Shortcut

Images receive class-dependent visual shortcuts/patches while labels remain unchanged.

The shortcut mechanism is class-dependent so that the model can learn an artificial visual cue.

Tests verify:

- healthy data remains unchanged
- label noise is deterministic
- requested noise fraction changes
- class imbalance removes/reduces selected classes
- shortcut modifies images but not labels
- shortcut mapping is class-dependent

Result:

```text
5/5 fault-injection tests passed
```

---

## 4.4 Behavioural probes

Implemented probes for:

- clean accuracy
- faulty/modified-data behaviour
- class-dependent patch probing
- mismatched patch probing
- corner attention
- related behavioural measurements used by feature extraction

The patch probe checks whether a model follows the injected visual shortcut.

The corner-attention measurement is used as additional evidence for shortcut behaviour.

Result:

```text
3/3 probe tests passed
```

---

# 5. ORIGINAL MODEL.PY ERROR — FIXED

During development, `src/model.py` contained accidentally appended Python code:

```text
return self.classifier(features)from __future__ import annotations
```

This caused:

```text
SyntaxError: invalid syntax
```

during pytest collection.

The problem was not related to PyTorch or the model architecture. It was a source-file corruption/duplication issue.

The file was corrected and the complete model test suite subsequently passed.

---

# 6. LEVEL 2 — CONTROLLED MODEL ZOO

## Status: COMPLETE

A reproducible model-zoo generation system was implemented.

The zoo contains:

```text
100 trained models
```

covering the four controlled conditions.

The training system supports resuming from existing checkpoints so that long-running training can survive interruptions.

### Preflight

The required small preflight run was performed successfully:

```text
4/4 models
```

### Full zoo

Final result:

```text
100/100 checkpoints
```

The zoo completed successfully.

---

## Training robustness

During the long run, there was an environment/filesystem interruption while writing one checkpoint.

The incomplete checkpoint was removed and the resumable training process restarted.

The final result was successfully validated as:

```text
100/100 models present
```

No incomplete checkpoint was knowingly retained.

---

# 7. LEVEL 3 — FEATURE EXTRACTION

## Status: COMPLETE

A feature extraction pipeline was implemented over the complete 100-model zoo.

The extracted features represent model behaviour rather than training metadata.

The feature table contains:

```text
100 rows
14 columns
```

and was validated for:

- required feature columns
- unique feature columns
- no NaN values
- no infinity values
- correct feature-frame contract

---

## Important behavioural result

The shortcut models showed a strong difference between matched and mismatched patch behaviour.

Observed aggregate result:

```text
Matched-patch accuracy:    ~98.4%
Mismatched-patch accuracy: ~35.9%
```

This provides a measurable behavioural signal that shortcut models respond to the injected visual cue.

---

## Entropy validation correction

During validation, entropy was initially checked against an incorrect `[0,1]` range.

For a 10-class prediction distribution, entropy can reach:

```text
ln(10) ≈ 2.303
```

The validation bound was corrected and feature extraction was rerun.

The final feature table passed validation.

---

# 8. LEVEL 4 — MODEL DOCTOR

## Status: COMPLETE

The diagnostic classifier ("Model Doctor") was trained using the extracted behavioural features.

The primary Doctor model is:

```text
Random Forest
```

The evaluation was designed to avoid obvious leakage from metadata columns.

Tests specifically verify that leakage columns are not used as model features.

---

## Doctor results

### Random Forest holdout

```text
Accuracy: 1.00
F1:       1.00
```

### 5-fold cross-validation

```text
0.98 ± 0.0245
```

### Logistic Regression comparison

```text
0.95
```

### Dummy baseline

```text
0.25
```

The dummy baseline is consistent with a four-class classification problem under the implemented baseline.

---

## Label-shuffling sanity check

Because the holdout performance was perfect, a shuffled-label sanity check was also performed.

Result:

```text
0.34 ± 0.139
```

This did not reproduce the original near-perfect CV performance, providing an additional sanity check against the result being purely arbitrary.

---

# 9. LEVEL 5A — DIAGNOSIS + EXPLAINABILITY + STATIC EXPORT

## Status: COMPLETE

The real diagnosis pipeline was implemented.

Given a trained model, the system can produce:

```text
diagnosis
confidence
probabilities
diagnostic features
ground truth
suggested fix
corner attention
Grad-CAM evidence
```

---

## Diagnosis contract

The diagnosis JSON follows the required API/data contract.

A real end-to-end diagnosis was tested using:

```text
model_0003
```

and the Doctor predicted:

```text
shortcut
```

with the expected behavioural evidence.

---

## Static exports

The complete static diagnosis export was generated for:

```text
100/100 models
```

This includes diagnosis JSON outputs and Grad-CAM assets used by the frontend.

---

# 10. GRAD-CAM

## Status: COMPLETE

Grad-CAM support was implemented and tested.

Tests verify:

- Grad-CAM shape
- valid value range
- RGB overlay output
- corner-score behaviour/range

Result:

```text
3/3 Grad-CAM tests passed
```

Grad-CAM outputs are included in the exported model diagnosis assets.

---

# 11. API / DIAGNOSIS CONTRACT

## Status: COMPLETE

The project includes a documented diagnosis contract.

Tests verify that the required fields are documented.

Result:

```text
1/1 API contract test passed
```

---

# 12. COMPLETE PYTHON TEST SUITE

## Current result

The complete automated Python suite currently reports:

```text
32 passed in 8.65s
```

Breakdown:

```text
API contract                  PASS
Data                          PASS
Diagnosis                     PASS
Doctor                        PASS
Feature extraction            PASS
Fault injection               PASS
Grad-CAM                      PASS
Model                         PASS
Probes                        PASS
Scaffold                      PASS
Training                      PASS
Model zoo specification       PASS
```

Overall:

```text
32/32 tests passed
```

---

# 13. LEVEL 5B — REACT FRONTEND

## Status: IN PROGRESS / CORE UI WORKING

A React + Vite frontend has been implemented.

The frontend includes:

```text
Header
Model selector
Diagnosis card
Probability bars
Feature table
Grad-CAM gallery
Suggested fix
Ground-truth badge
Corner-attention display
```

---

## Frontend dependency installation

Initially PowerShell blocked:

```text
npm.ps1
```

because of the local Windows execution policy.

Instead of changing the system execution policy, the project was run using:

```powershell
npm.cmd install
```

This succeeded.

Result:

```text
65 packages added
66 packages audited
0 vulnerabilities
```

---

## Frontend production build

The Vite build succeeded:

```text
vite v7.3.6
33 modules transformed
built successfully in 1.09s
```

Therefore the frontend source compiles successfully.

---

## Frontend development server

The development server successfully starts with:

```powershell
npm.cmd run dev
```

and is available at:

```text
http://localhost:5173/
```

---

# 14. FRONTEND RUNTIME BUG — FIXED

The first frontend launch showed a completely white screen.

Browser console revealed:

```text
Uncaught ReferenceError: React is not defined
```

The cause was that JSX components were using `React` without importing it under the current Vite configuration.

`App.jsx` was fixed by adding:

```jsx
import React from "react";
```

The same fix was applied to the JSX components:

```text
Header.jsx
ModelSelector.jsx
DiagnosisCard.jsx
ProbabilityBars.jsx
FeatureTable.jsx
GradcamGallery.jsx
SuggestedFix.jsx
GroundTruthBadge.jsx
```

After the fix, the React UI rendered successfully.

---

# 15. FRONTEND END-TO-END CHECK ALREADY VERIFIED

The frontend currently successfully displays a real model diagnosis.

Tested model:

```text
model_0001 · class_imbalance
```

The UI displayed:

```text
Diagnosis:
Class Imbalance

Confidence:
100.0%

Ground truth:
class imbalance
```

The probability section showed:

```text
class imbalance    100.0%
healthy              0.0%
label noise          0.0%
shortcut             0.0%
```

The diagnostic evidence section also rendered.

Therefore:

```text
Static diagnosis data
        ↓
React frontend
        ↓
Model selection
        ↓
Diagnosis display
```

is currently working.

---

# 16. CURRENT FRONTEND STATUS

### Confirmed working

- React app starts
- Vite development server starts
- Page renders
- Header renders
- Model selector renders
- Model data loads
- Diagnosis loads
- Ground truth loads
- Probability bars render
- Diagnostic evidence renders

### Still needs final validation

Test at least one model from each class:

```text
healthy
class_imbalance
label_noise
shortcut
```

For each one verify:

```text
Doctor diagnosis
Ground truth
probabilities
features
Grad-CAM
suggested fix
corner attention
```

---

# 17. LEVEL 6 — FINALIZATION

## Status: PENDING

The remaining Level 6 work is:

### Technical

- Complete frontend validation for all four fault types
- Validate Grad-CAM rendering in the browser
- Validate all static asset paths
- Check responsive layout
- Check browser console for runtime warnings/errors
- Verify production build after final frontend changes
- Run the complete Python test suite again after final changes
- Perform an end-to-end demonstration from a clean project directory

### Documentation

Finalize:

```text
README.md
PROGRESS.md
methodology
architecture
experimental setup
feature definitions
evaluation methodology
results
limitations
future scope
reproducibility instructions
```

### Demonstration

Prepare a clean demo sequence:

```text
1. Open Model Doctor
2. Select a model
3. Show known ground truth
4. Show Doctor diagnosis
5. Show confidence/probabilities
6. Show behavioural evidence
7. Show Grad-CAM
8. Show suggested fix
9. Switch to another failure mode
10. Explain how the evidence changes
```

### Academic presentation

Prepare:

- problem statement
- motivation
- objectives
- novelty
- architecture
- methodology
- dataset
- fault injection
- behavioural probes
- feature extraction
- Doctor model
- evaluation
- explainability
- limitations
- future work
- conclusion

### Viva preparation

Prepare answers for:

- Why MNIST?
- Why controlled fault injection?
- Why use a second model?
- Why Random Forest?
- What features are used?
- How is leakage avoided?
- How do you know the diagnosis is correct?
- What is shortcut learning?
- What does Grad-CAM contribute?
- Why not simply inspect model accuracy?
- What happens on a failure mode not represented in training?
- What are the limitations of the current system?
- How could this be extended to real-world image classifiers?

---

# 18. CURRENT PROJECT STATE

## Completed

```text
[✓] Project scaffold
[✓] MNIST data pipeline
[✓] TinyCNN
[✓] Controlled fault injection
[✓] Behavioural probes
[✓] 100-model zoo
[✓] Feature extraction
[✓] Random Forest Doctor
[✓] Doctor evaluation
[✓] Leakage checks
[✓] Label-shuffle sanity check
[✓] Diagnosis pipeline
[✓] Diagnosis contract
[✓] Grad-CAM
[✓] 100 static diagnosis exports
[✓] React frontend source
[✓] npm installation
[✓] Vite production build
[✓] Vite development server
[✓] React runtime bug fixed
[✓] Frontend successfully rendering
[✓] At least one complete UI diagnosis verified
```

## Remaining

```text
[ ] Validate all four fault types in UI
[ ] Validate Grad-CAM browser rendering
[ ] Validate all static asset paths
[ ] Final frontend polish
[ ] Final full regression test
[ ] Final README
[ ] Final PROGRESS.md
[ ] Final architecture/workflow documentation
[ ] Final demo script
[ ] Final presentation
[ ] Final viva preparation
[ ] Clean-install/reproducibility verification
[ ] Final 100% completion review
```

---

# 19. IMPORTANT VERIFIED RESULTS

These are the main quantitative results currently available:

| Experiment | Result |
|---|---:|
| Python automated tests | **32/32 passed** |
| Model zoo | **100/100 models** |
| Feature rows | **100** |
| Feature columns | **14** |
| Random Forest holdout accuracy | **1.00** |
| Random Forest holdout F1 | **1.00** |
| Random Forest 5-fold CV | **0.98 ± 0.0245** |
| Logistic Regression comparison | **0.95** |
| Dummy baseline | **0.25** |
| Shuffled-label CV | **0.34 ± 0.139** |
| Shortcut matched-patch accuracy | **~98.4%** |
| Shortcut mismatched-patch accuracy | **~35.9%** |
| Static diagnosis exports | **100/100** |
| Grad-CAM | **Implemented + tested** |
| Frontend production build | **Successful** |

---

# 20. CURRENT COMPLETION ESTIMATE

The core research/engineering implementation is substantially complete.

The remaining work is primarily:

```text
frontend validation
+
integration testing
+
documentation
+
presentation/demo preparation
+
final reproducibility check
```

The project should **not yet be marked 100% complete** until those remaining checks are performed.

---

# 21. NEXT IMMEDIATE STEP

The next task is:

### Test the four diagnosis categories in the live React UI.

Use the model selector and verify:

```text
Healthy
Class Imbalance
Label Noise
Shortcut
```

For each model record:

```text
Model ID
Ground truth
Doctor diagnosis
Confidence
Most important evidence
Grad-CAM visible?
Suggested fix visible?
```

After that, proceed to the final integration and Level 6 completion work.

---

## FINAL TARGET

The project is considered complete only when:

```text
ML pipeline
      +
100-model experimental dataset
      +
Doctor classifier
      +
explainability
      +
working React demo
      +
reproducible setup
      +
tests
      +
documentation
      +
presentation
      +
viva preparation
```

have all been validated.

**Current state: Core Model Doctor system is working locally; final integration and project packaging remain.**
