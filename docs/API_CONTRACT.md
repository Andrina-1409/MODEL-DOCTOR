# Model Doctor Diagnosis API Contract

The central diagnosis function is:

```python
diagnose(model_id: str) -> dict
```

It is UI-independent and the diagnosis is produced by the trained Random Forest Model Doctor.

## Response

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

## Field meanings

- `model_id`: unique model identifier.
- `diagnosis`: Random Forest prediction. One of `healthy`, `class_imbalance`, `label_noise`, `shortcut`.
- `confidence`: highest Random Forest class probability.
- `probabilities`: probability for every doctor class.
- `features`: the ten measurable diagnostic features.
- `gradcam_images`: paths to generated Grad-CAM visualizations.
- `corner_attention`: maximum fraction of Grad-CAM mass in a 6x6 image corner.
- `suggested_fix`: fixed explanatory mapping associated with the diagnosis.
- `ground_truth`: experimental fault metadata from the controlled model zoo. It is reported for evaluation/demo transparency and is never used as a doctor input feature.

## Fixed suggested-fix mapping

| Diagnosis | Suggested fix |
|---|---|
| healthy | no fault detected |
| class_imbalance | rebalance the data or use a weighted loss |
| label_noise | clean the labels or use a robust loss |
| shortcut | remove or randomize the spurious cue and use augmentation |

The JSON contract must be updated if the response structure changes.
