from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnose import diagnose


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static React demo data.")
    parser.add_argument("--model-dir", default="models/zoo")
    parser.add_argument("--output-dir", default="frontend/public/data")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(model_dir.glob("model_*.pt"))
    if len(checkpoints) != 100:
        raise RuntimeError(f"Expected 100 checkpoints, found {len(checkpoints)}")

    index = []
    for checkpoint in checkpoints:
        model_id = checkpoint.stem
        print(f"[export] {model_id}")
        result = diagnose(model_id, model_dir=model_dir, output_dir=output_dir / "gradcam")
        result["gradcam_images"] = [
            path.replace("frontend/public/", "").replace("\\", "/")
            for path in result["gradcam_images"]
        ]
        destination = output_dir / f"{model_id}.json"
        destination.write_text(json.dumps(result, indent=2))
        index.append({
            "model_id": model_id,
            "fault": result["ground_truth"],
        })

    (output_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Exported {len(index)} demo models.")


if __name__ == "__main__":
    main()
