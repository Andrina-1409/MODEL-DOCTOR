# Model Doctor — Report Outline

## 1. Abstract
Describe Model Doctor as a controlled experiment that diagnoses four MNIST CNN behaviours from measurable evidence.

## 2. Problem Statement
Manual ML debugging often requires inspecting multiple behavioural signals. This project tests whether a meta-classifier can identify known failure modes.

## 3. Fault Taxonomy
- Healthy
- Class imbalance
- Label noise
- Shortcut learning

Explain each injection and its expected behavioural signature.

## 4. Dataset and Reproducibility
- MNIST
- 20,000 training images
- 10,000 clean test images
- deterministic subset seed
- model seeds
- 5 epochs, batch size 256, Adam, learning rate 1e-3

## 5. TinyCNN
Document the architecture and exposed feature representations.

## 6. Model Zoo
Document the 100-model design, fault balance, severity range, checkpoint metadata, and resumable training.

## 7. Diagnostic Features
Explain all ten features and why each is measurable without ground-truth fault metadata.

## 8. Model Doctor
Random Forest with 300 estimators and the model-level 80/20 stratified split.

## 9. Evaluation
Insert only values from `reports/metrics.json`. Include Random Forest, Logistic Regression, dummy baseline, 5-fold CV, confusion matrix, and feature importance.

## 10. Grad-CAM
Explain manual hook-based Grad-CAM on the final convolutional layer and corner-attention analysis.

## 11. Static Demo
Explain React static JSON flow.

## 12. Limitations
Discuss synthetic faults, small CNN architecture, MNIST scope, and controlled experimental assumptions.

## 13. Future Work
Possible real-world datasets, additional architectures, richer faults, FastAPI, and Spring Boot integration.

## 14. Conclusion
Summarize only conclusions supported by the executed experiments.
