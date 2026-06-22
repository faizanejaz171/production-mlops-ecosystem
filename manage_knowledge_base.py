import chromadb
from sentence_transformers import SentenceTransformer


def setup_and_query_db():
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    collection = chroma_client.get_or_create_collection(
        name="testing_guidelines", metadata={"hnsw:space": "cosine"}
    )

    guidelines = [
        "If bounding box area is zero or microscopic, reject the file and route to rejected_dataset.",
        "Confidence threshold for a valid object detection must be strictly above 0.65.",
        "Annotators must draw tight bounding boxes around the full visible body of human_worker.",
        "For blurred images, tag with low_quality attribute but do not reject if target is visible.",
        "Red uniforms represent company employees, while blue jackets are for visitors.",
        "When overlapping occurs, create separate bounding boxes for each individual object.",
        "Missing labels on prominent assets must be marked as Critical Annotation Error.",
        "Always verify that the coordinates x_center and y_center are normalized between 0 and 1.",
        "If an asset is occluded more than 80 percent, do not tag it, skip the annotation.",
        "Every batch validation report must be automatically synced to the company Slack channel.",
    ]

    ids = [f"id_{i}" for i in range(len(guidelines))]

    embeddings = model.encode(guidelines, convert_to_tensor=False)
    embeddings_list = [emb.tolist() for emb in embeddings]

    collection.add(documents=guidelines, embeddings=embeddings_list, ids=ids)

    query_text = "What should I do if the detection confidence is very low?"
    query_vector = model.encode(query_text).tolist()

    results = collection.query(query_embeddings=[query_vector], n_results=3)

    return results


if __name__ == "__main__":
    print("🚀 Initializing Local Persistent ChromaDB Knowledge Base...")
    output = setup_and_query_db()

    print("\n" + "═" * 60)
    print("🔍 CHROMADB SEMANTIC SEARCH RESULTS")
    print("═" * 60)

    if output:
        documents = output["documents"][0]
        distances = output["distances"][0]
        ids = output["ids"][0]

        for doc, dist, doc_id in zip(documents, distances, ids):
            score = 1.0 - dist
            print(f"🆔 ID       : {doc_id}")
            print(f"📄 Match    : {doc}")
            print(f"📊 Similarity Score: {score:.4f}")
            print("─" * 60)
    print("═" * 60 + "\n")
