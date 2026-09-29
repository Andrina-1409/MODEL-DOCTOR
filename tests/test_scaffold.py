"""Level 0 project scaffold checks."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_level_zero_required_directories_exist() -> None:
    """The directories required by the Level 0 specification are present."""
    required_directories = (
        "src",
        "frontend",
        "models",
        "features",
        "reports",
        "docs",
        "tests",
        "notebooks",
    )

    missing = [
        directory
        for directory in required_directories
        if not (PROJECT_ROOT / directory).is_dir()
    ]

    assert not missing, f"Missing Level 0 directories: {missing}"


def test_level_zero_required_files_exist() -> None:
    """The Level 0 dependency and Git configuration files are present."""
    required_files = ("requirements.txt", ".gitignore")

    missing = [
        file_name
        for file_name in required_files
        if not (PROJECT_ROOT / file_name).is_file()
    ]

    assert not missing, f"Missing Level 0 files: {missing}"
