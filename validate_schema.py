import json
from pydantic import BaseModel, ValidationError, Field


# Model updated with confidence constraints
class ModelTestResult(BaseModel):
    image_id: str
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)  # 0.0 se 1.0 ke beech hona chahiye
    ground_truth: str


if __name__ == "__main__":
    sample_json_data = '{"image_id": "IMG_4041", "predicted_class": "cat", "confidence": 0.98, "ground_truth": "cat"}'
    try:
        data_dict = json.loads(sample_json_data)
        validated_data = ModelTestResult(**data_dict)
        print("✅ Success: Data is perfectly valid!")
    except ValidationError:
        print("❌ Validation Failed!")
