# Model Doctor — Professor Demo Proof Library

## Purpose

The project now includes a curated six-case offline proof library. These cases are prevalidated Model Doctor checkpoints with all required diagnosis JSON and Grad-CAM assets already bundled, so a live demonstration does not require internet access, model hunting, dataset searching, or manual preprocessing.

### Offline proof cases

| Proof case | Model | Architecture | Dataset | Known condition | Demo purpose |
|---|---|---|---|---|---|
| Healthy CNN Baseline | model_0004 | TinyCNN | MNIST | healthy | Establish normal behaviour |
| Class Imbalance Case | model_0001 | TinyCNN | MNIST | class imbalance | Demonstrate minority-class failure |
| Label Noise Case | model_0002 | TinyCNN | MNIST | label noise | Demonstrate corrupted-label behaviour |
| Shortcut Learning Case | model_0003 | TinyCNN | MNIST | shortcut | Demonstrate spurious-cue dependence |
| Healthy Reproducibility Case | model_0008 | TinyCNN | MNIST | healthy | Show a second independent healthy seed |
| Shortcut Stress Case | model_0007 | TinyCNN | MNIST | shortcut | Show repeatability of shortcut diagnosis |

These six cases are the recommended **professor demo set**. They are deliberately self-contained and use the same validated diagnosis assets as the 100-model research experiment.

## Public external model references

The project also records public model sources that can be downloaded once, stored locally, and added as future supported proof packs. Do not claim these binaries are bundled unless they are actually present.

1. **ResNet-18 / CIFAR-10 — Hugging Face**
   - Model: `bhumong/resnet18-cifar10`
   - Dataset: CIFAR-10
   - Task: image classification
   - Public model card reports 10 classes and a PyTorch ResNet-18 implementation.
   - Source: https://huggingface.co/bhumong/resnet18-cifar10

2. **ResNet-18 / ImageNet-1K — TorchVision**
   - Official pretrained ResNet-18 weights
   - 11,689,512 parameters
   - 224×224 inference crop and ImageNet normalization documented by TorchVision.
   - Source: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

3. **MobileNetV3-Small / ImageNet-1K — TorchVision**
   - Official pretrained MobileNetV3-Small weights
   - 2,542,856 parameters
   - 224×224 inference crop and ImageNet normalization documented by TorchVision.
   - Source: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html

## Demo rule

During a college demonstration, use the six offline proof cases. If asked whether the system can analyze another supported model, explain that the current research validation is on TinyCNN/MNIST and that external model packs are an extension path. Do not claim arbitrary-model support without validating the model's architecture, preprocessing, dataset, and diagnostic compatibility.
