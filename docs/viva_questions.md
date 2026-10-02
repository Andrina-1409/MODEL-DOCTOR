# Model Doctor — Viva Questions

## Fundamentals
1. What problem does Model Doctor solve?
2. Why use MNIST?
3. Why use a small CNN?
4. What are the four fault classes?
5. Why is fault injection controlled?

## Data and training
6. Why are the train/test subsets fixed?
7. Why must test images remain clean for shortcut injection?
8. Why is the shortcut mapping fixed?
9. What does severity mean for each fault?
10. Why is the training process deterministic?

## Diagnostic features
11. What is `acc_min_class`?
12. Why can class-accuracy standard deviation reveal imbalance?
13. What does entropy measure?
14. Why is `conf_wrong_rate` useful?
15. What does the train/validation loss gap indicate?
16. Why compare matched and mismatched patches?

## Meta-classifier
17. Why Random Forest?
18. Why compare Logistic Regression and a dummy baseline?
19. Why is the split performed at model level?
20. Why must model IDs be disjoint between train and test?
21. What is feature leakage?
22. Why are fault and severity forbidden as input features?

## Explainability
23. How does Grad-CAM work?
24. Why use the final convolutional layer?
25. What does corner attention measure?
26. Why should Grad-CAM be treated as evidence rather than the diagnosis?

## System
27. What does `diagnose(model_id)` do?
28. Why is the diagnosis function independent of React?
29. How does the static frontend work?
30. What would you change for a real production system?

## Scientific integrity
31. What would suspiciously perfect accuracy make you investigate?
32. What are the limitations of using synthetic faults?
33. How would you validate the approach on another dataset?
