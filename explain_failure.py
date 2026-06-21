import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class AIExplanationSchema(BaseModel):
    summary_roman_urdu: str
    severity: str
    suggested_fix_english: str


def ask_llm_to_explain(raw_error_msg, file_details):
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
    )

    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is completely missing.")
        return None

    system_prompt = (
        "You are an expert MLOps QA Assistant.\n"
        "Look at the raw validation error and translate it into clear Roman Urdu for annotators.\n"
        "You must output strictly a JSON object with keys: summary_roman_urdu, severity (LOW, MEDIUM, HIGH), and suggested_fix_english."
    )

    user_content = f"File: {file_details}\nError: {raw_error_msg}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw_json_string = response.choices[0].message.content
        validated_output = AIExplanationSchema.model_validate_json(raw_json_string)
        return validated_output

    except Exception as e:
        print(f"Failed to communicate or parse: {e}")
        return None


if __name__ == "__main__":
    sample_error = "Bounding box area is abnormally tiny or zero (Bad Annotation)."
    sample_file = "20260422190001997982051_0001.txt"

    print("🤖 Querying Free Groq AI Explainer Engine...")
    result = ask_llm_to_explain(sample_error, sample_file)

    if result:
        print("\n" + "═" * 60)
        print("🧠 LLM STRUCTURED EXPLANATION REPORT")
        print("═" * 60)
        print(f"Severity Level : 🚨 {result.severity}")
        print(f"Urdu Summary   : {result.summary_roman_urdu}")
        print(f"How to Fix     : 🛠️ {result.suggested_fix_english}")
        print("═" * 60 + "\n")
