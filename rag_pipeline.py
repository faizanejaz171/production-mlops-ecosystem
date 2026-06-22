import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="testing_guidelines")


def retrieve(query, n=3):
    query_vector = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=n)
    return results["documents"][0] if results["documents"] else []


def generate_answer(query, chunks):
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
    )

    context = "\n---\n".join(chunks)

    system_prompt = (
        "You are an expert MLOps QA Assistant.\n"
        "Your task is to answer the user's question strictly based ONLY on the provided Context below.\n"
        "CRITICAL RULES:\n"
        "1. Answer in clear, concise, and professional English.\n"
        "2. If the answer cannot be found in the provided Context, reply exactly with: "
        "'I don't know based on available guidelines.' Do not invent or hallucinate anything.\n"
        f"Context:\n{context}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {e}"


def rag_query(question):
    print(f"\n❓ User Question: {question}")

    retrieved_chunks = retrieve(question, n=2)
    print("📋 Retrieved Context Chunks from ChromaDB:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  [{i+1}] {chunk}")

    answer = generate_answer(question, retrieved_chunks)
    print(f"🤖 AI Grounded Answer:\n{answer}")
    print("═" * 60)


if __name__ == "__main__":
    print("🚀 Running End-to-End English RAG Validation Pipeline...")
    print("═" * 60)

    test_questions = [
        "What is the required confidence threshold for a valid object detection?",
        "What should I do if the bounding box area is zero or microscopic?",
        "What are the uniform color rules for company employees and visitors?",
        "What is the current monthly salary package for a senior MLOps Engineer?",
    ]

    for q in test_questions:
        rag_query(q)
