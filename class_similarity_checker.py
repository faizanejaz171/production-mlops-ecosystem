from sentence_transformers import SentenceTransformer, util


def check_class_similarity(classes_list):
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    embeddings = model.encode(classes_list, convert_to_tensor=True)

    similarity_pairs = []
    num_classes = len(classes_list)

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            score = util.cos_sim(embeddings[i], embeddings[j]).item()
            similarity_pairs.append((classes_list[i], classes_list[j], score))

    similarity_pairs.sort(key=lambda x: x[2], reverse=True)

    return similarity_pairs[:5]


if __name__ == "__main__":
    sample_classes = [
        "person",
        "human_worker",
        "red_car",
        "maroon_truck",
        "delivery_van",
        "office_chair",
        "visitor_badge",
    ]

    print("🤖 Local Embedding Model Loading & Processing...")
    top_confusions = check_class_similarity(sample_classes)

    print("\n" + "═" * 60)
    print("🧠 EMBEDDING-BASED CLASS SIMILARITY REPORT")
    print("═" * 60)

    for class_a, class_b, score in top_confusions:
        print(f"🔄 Pairs: [{class_a}]  🟢  [{class_b}]")
        print(f"📊 Semantic Similarity Score: {score:.4f}")

        if score > 0.60:
            print(
                "⚠️  Warning: High semantic similarity! Annotators might get confused."
            )
        print("─" * 60)

    print("═" * 60 + "\n")
