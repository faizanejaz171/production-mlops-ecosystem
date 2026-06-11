import json
from pydantic import BaseModel, ValidationError

# 1. Schema Define Kiya
class ModelTestResult(BaseModel):
    image_id: str
    predicted_class: str
    confidence: float
    ground_truth: str

# 2. Dummy Sample Data (Jo kal ko tumhari office pipeline se aayega)
sample_json_data = """
{
    "image_id": "IMG_4041",
    "predicted_class": "cat",
    "confidence": 0.98,
    "ground_truth": "cat"
}
"""

if __name__ == "__main__":
    try:
        # JSON string ko parse karke dictionary banaya
        data_dict = json.loads(sample_json_data)
        
        # Pydantic model ke zariye data validate kiya
        validated_data = ModelTestResult(**data_dict)
        
        print("✅ Success: Data is perfectly valid!")
        print(validated_data)
        
    except ValidationError as e:
        print("❌ Validation Failed! Errors found:")
        print(e.json(indent=2))
