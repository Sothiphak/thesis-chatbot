"""
calibrate_relevance.py

Run this ON YOUR MACHINE, against your real milvus_rag.db, before trusting
RELEVANCE_THRESHOLD in app.py. I cannot calibrate this myself -- I don't
have access to your actual embedded documents, only a synthetic test I built
separately to confirm the field name and score direction (distance, higher =
more similar, assuming COSINE/IP metric).

What this does: runs a handful of clearly IN-SCOPE and clearly OUT-OF-SCOPE
queries against your real collection, prints the actual top-match score for
each. Look at the gap between the two clusters of scores and pick a
threshold that sits between them.
"""

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

DB_PATH = "milvus_rag.db"
COLLECTION_NAME = "hr_it_policies"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Edit these to match what's ACTUALLY in your 15 documents -- these are
# generic guesses at plausible HR/IT topics, not verified against your real
# knowledge base content. Swap in real questions you know the answer to.
IN_SCOPE_QUERIES = [
    "តើថ្ងៃឈប់សម្រាកប្រចាំឆ្នាំមានប៉ុន្មានថ្ងៃ?",   # annual leave days
    "What is the password rotation policy?",
    "តើត្រូវធ្វើដូចម្តេចប្រសិនបើសង្ស័យអ៊ីមែលបន្លំ?",  # suspected phishing email
    "How many days of paid sick leave do I get?",
    "What is the code of conduct?",
]

# Clearly unrelated to any HR/IT policy document -- if the gate is working,
# these should score noticeably LOWER than the in-scope queries above.
OUT_OF_SCOPE_QUERIES = [
    "តើអាហារខ្មែរប្រពៃណីមួយណាដែលអ្នកគិតថាល្អបំផុត?",  # best traditional Khmer food
    "What's the weather like today?",
    "Can you write me a poem about the ocean?",
    "តើប្រវត្តិសាស្ត្រកម្ពុជាចាប់ផ្តើមនៅឆ្នាំណា?",  # when does Cambodian history begin
    "What is the capital of France?",
]


def main():
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to {DB_PATH}...")
    client = MilvusClient(DB_PATH)
    client.load_collection(COLLECTION_NAME)

    print("\nVerify metric type is COSINE or IP (NOT L2) before trusting these numbers:")
    try:
        print(client.describe_collection(COLLECTION_NAME))
    except Exception as e:
        print(f"Could not fetch automatically: {e}")
    print()

    def top_score(query: str) -> float:
        emb = embed_model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
        results = client.search(COLLECTION_NAME, data=[emb], limit=1, output_fields=["text", "source"])
        if not results or not results[0]:
            return 0.0
        return results[0][0]["distance"]

    print("=" * 70)
    print("IN-SCOPE queries (expect HIGHER scores):")
    in_scope_scores = []
    for q in IN_SCOPE_QUERIES:
        s = top_score(q)
        in_scope_scores.append(s)
        print(f"  {s:.4f}  {q}")

    print("\nOUT-OF-SCOPE queries (expect LOWER scores):")
    out_scope_scores = []
    for q in OUT_OF_SCOPE_QUERIES:
        s = top_score(q)
        out_scope_scores.append(s)
        print(f"  {s:.4f}  {q}")

    print("=" * 70)
    min_in = min(in_scope_scores)
    max_out = max(out_scope_scores)
    print(f"Lowest in-scope score:  {min_in:.4f}")
    print(f"Highest out-of-scope score: {max_out:.4f}")

    if min_in > max_out:
        suggested = (min_in + max_out) / 2
        print(f"\nClean separation. Suggested RELEVANCE_THRESHOLD ~= {suggested:.4f}")
        print("(midpoint between the two clusters -- adjust toward min_in if you want to")
        print(" be more permissive, toward max_out if you want to be stricter)")
    else:
        print("\nWARNING: scores overlap -- no clean separation between in-scope and")
        print("out-of-scope queries. A single threshold may not cleanly distinguish them.")
        print("Consider: more/better in-scope test queries, checking metric type is right,")
        print("or accepting some false positives/negatives as a known limitation.")


if __name__ == "__main__":
    main()
