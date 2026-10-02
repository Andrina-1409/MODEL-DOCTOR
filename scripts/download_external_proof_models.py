"""Download public reference models for future external proof packs.

This script intentionally does not claim arbitrary-model diagnosis. The current
validated live demo uses proof_models/checkpoints. External models need an
adapter, dataset, preprocessing metadata, and compatible probes before they
can be diagnosed by the Model Doctor.
"""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "proof_models" / "external"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "resnet18_imagenet1k.pth": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "mobilenet_v3_small_imagenet1k.pth": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}

for filename, url in SOURCES.items():
    destination = OUT / filename
    print(f"Downloading {filename} ...")
    urlretrieve(url, destination)
    print(f"Saved {destination}")

print("External weights downloaded. These are reference models only until an adapter + compatible dataset/probes are configured.")
