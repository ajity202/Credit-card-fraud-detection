from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist():

    required_directories = [
        "api",
        "app",
        "configs",
        "data",
        "models",
        "notebooks",
        "reports",
        "src",
        "tests"
    ]

    for directory in required_directories:

        path = PROJECT_ROOT / directory

        assert path.exists(), (
            f"Missing directory: {directory}"
        )


def test_required_files_exist():

    required_files = [
        "api/main.py",
        "app/streamlit_app.py",
        "configs/config.yaml",
        "src/fraud_detection/config.py",
        "src/fraud_detection/data.py",
        "src/fraud_detection/models.py",
        "src/fraud_detection/__init__.py",
        "requirements.txt",
        "README.md",
        "Dockerfile"
    ]

    for file in required_files:

        path = PROJECT_ROOT / file

        assert path.exists(), (
            f"Missing file: {file}"
        )