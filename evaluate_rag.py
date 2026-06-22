import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rag_pipeline import retrieve, generate_answer

load_dotenv()


def llm_as_a_judge(question, context, answer, ground_truth):
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
    )

    judge_prompt = (
        "You are an objective AI Evaluation Judge assessing a RAG pipeline quality.\n"
        "Analyze the inputs and output a valid JSON object with exactly these 3 keys and float scores between 0.0 and 1.0:\n"
        "1. 'faithfulness': (Is the answer 100% derived ONLY from the context? No outside knowledge = 1.0, otherwise lower)\n"
        "2. 'answer_relevancy': (Does the answer directly address the specific question? Highly relevant = 1.0)\n"
        "3. 'context_recall': (Does the retrieved context contain the information matching the ground_truth reference? Full match = 1.0)\n\n"
        "Output ONLY the raw valid JSON object. No conversational filler text."
    )

    user_content = (
        f"Question: {question}\n"
        f"Retrieved Context: {context}\n"
        f"Generated Answer: {answer}\n"
        f"Ground Truth Reference: {ground_truth}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Judge Communication Error: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}


if __name__ == "__main__":
    print("📋 Setting Up Manual Evaluation Dataset & Ground Truth...")
    print("═" * 60)

    # Office Guidelines Ground Truth Dataset
    eval_dataset = [
        {
            "question": "What is the required confidence threshold for a valid object detection?",
            "ground_truth": "The confidence threshold must be strictly above 0.65 for a valid detection.",
        },
        {
            "question": "What should I do if the bounding box area is zero or microscopic?",
            "ground_truth": "Reject the file completely and route it to the rejected_dataset folder.",
        },
        {
            "question": "What are the uniform color rules for company employees and visitors?",
            "ground_truth": "Red uniforms represent company employees, while blue jackets are reserved for visitors.",
        },
    ]

    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_recall = 0.0
    count = len(eval_dataset)

    print(f"🤖 Processing and Evaluating {count} System Test Cases...\n")

    for i, row in enumerate(eval_dataset):
        q = row["question"]
        gt = row["ground_truth"]

        # 1. Pipeline Execution
        chunks = retrieve(q, n=2)
        joined_context = " ".join(chunks)
        ans = generate_answer(q, chunks)

        # 2. Judge Evaluation
        scores = llm_as_a_judge(q, joined_context, ans, gt)

        print(f"📝 Test Case {i+1}:")
        print(f"  ❓ Q : {q}")
        print(
            f"  📊 Scores -> Faithfulness: {scores.get('faithfulness')} | Relevancy: {scores.get('answer_relevancy')} | Recall: {scores.get('context_recall')}"
        )
        print("─" * 60)

        total_faithfulness += scores.get("faithfulness", 0.0)
        total_relevancy += scores.get("answer_relevancy", 0.0)
        total_recall += scores.get("context_recall", 0.0)

    # 3. Compute Baseline Metrics
    print("\n" + "═" * 60)
    print("📊 FINAL RAGAS BASELINE PERFORMANCE REPORT")
    print("═" * 60)
    print(f"📈 Average Faithfulness    : {total_faithfulness/count:.4f}")
    print(f"📈 Average Answer Relevancy : {total_relevancy/count:.4f}")
    print(f"📈 Average Context Recall   : {total_recall/count:.4f}")
    print("═" * 60 + "\n")
