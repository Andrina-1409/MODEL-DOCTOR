import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_professor_proof_library_has_six_cases():
    catalog = json.loads((ROOT / "frontend/public/proof/catalog.json").read_text())
    assert len(catalog) == 6
    assert {item["fault"] for item in catalog} == {"healthy", "class_imbalance", "label_noise", "shortcut"}
    for item in catalog:
        assert (ROOT / "models/zoo" / f"{item['model_id']}.pt").exists()
        assert (ROOT / "frontend/public/data" / f"{item['model_id']}.json").exists()


def test_external_catalog_is_reference_only():
    catalog = json.loads((ROOT / "proof_models/external_catalog.json").read_text())
    assert len(catalog) >= 3
    assert all(item["status"] == "external_reference" for item in catalog)
