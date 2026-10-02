from pathlib import Path


def test_api_contract_documents_required_fields():
    text = Path("docs/API_CONTRACT.md").read_text()
    for field in [
        "model_id", "diagnosis", "confidence", "probabilities",
        "features", "gradcam_images", "corner_attention",
        "suggested_fix", "ground_truth",
    ]:
        assert field in text
