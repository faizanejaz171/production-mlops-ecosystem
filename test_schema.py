import pytest
from pydantic import ValidationError
from validate_schema import ModelTestResult


def test_valid_data():
    # Sahi data paas hona chahiye
    data = {
        "image_id": "IMG_1",
        "predicted_class": "dog",
        "confidence": 0.85,
        "ground_truth": "dog",
    }
    assert ModelTestResult(**data).confidence == 0.85


def test_invalid_confidence():
    # Confidence > 1.0 par ValidationError aani chahiye
    invalid_data = {
        "image_id": "IMG_1",
        "predicted_class": "dog",
        "confidence": 1.5,
        "ground_truth": "dog",
    }
    with pytest.raises(ValidationError):
        ModelTestResult(**invalid_data)


def test_invalid_data_type():
    # Data type galat hone par ValidationError aani chahiye
    bad_type_data = {
        "image_id": "IMG_1",
        "predicted_class": "dog",
        "confidence": "high_confidence",
        "ground_truth": "dog",
    }
    with pytest.raises(ValidationError):
        ModelTestResult(**bad_type_data)
