# Model Doctor — PPT Outline

1. **Title** — Model Doctor: Diagnosing Why ML Models Fail
2. **Problem** — Model debugging is often manual and signal-heavy.
3. **Idea** — Build a controlled model clinic with known faults.
4. **Faults** — Healthy, class imbalance, label noise, shortcut.
5. **Architecture** — MNIST → fault injection → 100 CNNs → features → Random Forest → diagnosis → Grad-CAM → React.
6. **TinyCNN** — Small, inspectable architecture.
7. **Diagnostic Features** — Ten behavioural measurements.
8. **Model Doctor** — Random Forest, no metadata leakage.
9. **Evaluation** — Insert actual metrics after Level 4.
10. **Evidence** — Confusion matrix, feature importance, Grad-CAM.
11. **Demo** — Select model and inspect diagnosis/evidence/fix.
12. **Scientific Integrity** — Ground truth is never a doctor input.
13. **Limitations** — Controlled MNIST experiment.
14. **Conclusion** — Use only observed results.
