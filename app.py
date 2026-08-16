"""
RAG Chat Interface — CLUSTER-DEPLOYABLE VERSION

This is the same app as rag_chat_interface.py, adapted to run *inside*
OpenShift instead of on your laptop. The one real change: it connects to
vLLM via the cluster's internal service address instead of localhost,
since there's no port-forward once this is running as a pod itself.

This file is meant to be pushed to a git repo and deployed via:
    oc new-app python:3.12~<your-repo-url>
    oc expose svc/<app-name>

The milvus_rag.db file (already built) must be committed to the SAME
repo, in the same folder as this file, so it's included in the build.
"""

import os
import gradio as gr
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

DB_PATH = "milvus_rag.db"
COLLECTION_NAME = "hr_it_policies"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TOP_K = 3

# Internal cluster address — works because vLLM and this app can reach
# each other across namespaces by default (only tenant-a/tenant-b have
# restrictive NetworkPolicy applied). Format is:
# <service-name>.<namespace>.svc.cluster.local:<port>
VLLM_INTERNAL_URL = os.environ.get(
    "VLLM_URL", "http://vllm-sealion.model-serving.svc.cluster.local:8000/v1"
)
MODEL_NAME = os.environ.get("MODEL_NAME", "sealion")

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Connecting to knowledge base at {DB_PATH}...")
milvus_client = MilvusClient(DB_PATH)
milvus_client.load_collection(COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' loaded and ready.")

print(f"Connecting to vLLM at {VLLM_INTERNAL_URL}...")
vllm_client = OpenAI(base_url=VLLM_INTERNAL_URL, api_key="not-needed")


def retrieve(query: str, top_k: int = TOP_K):
    query_embedding = embed_model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "source"],
    )
    return results[0]


def build_augmented_prompt(query: str, retrieved_chunks):
    context_blocks = []
    sources = []
    for hit in retrieved_chunks:
        text = hit["entity"]["text"]
        source = hit["entity"]["source"]
        context_blocks.append(f"[Source: {source}]\n{text}")
        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are an internal company assistant. Answer the employee's question "
        "using ONLY the information in the provided company documents below. "
        "If the documents don't contain the answer, say so honestly rather than "
        "guessing. Always answer in the same language the question was asked in. "
        "At the end of your answer, briefly cite which document(s) you used.\n\n"
        f"=== COMPANY DOCUMENTS ===\n{context}\n=== END DOCUMENTS ==="
    )
    return system_prompt, sources


def chat(message, history):
    retrieved = retrieve(message)
    system_prompt, sources = build_augmented_prompt(message, retrieved)

    messages = [{"role": "system", "content": system_prompt}]
    for item in history:
        if isinstance(item, dict):
            messages.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    try:
        response = vllm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=400,
            extra_body={"repetition_penalty": 1.15},
        )
        answer = response.choices[0].message.content
        source_note = f"\n\n📄 *Retrieved from: {', '.join(sources)}*"
        return answer + source_note
    except Exception as e:
        return f"⚠️ Could not reach the model server. Error: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="Sovereign Khmer RAG Chatbot",
    description=(
        "Internal employee assistant, grounded in company HR/IT policy documents. "
        "Powered by aisingapore/Llama-SEA-LION-v3-8B-IT (INT4, vLLM), running entirely "
        "on institution-controlled infrastructure."
    ),
    examples=[
        "តើថ្ងៃឈប់សម្រាកប្រចាំឆ្នាំមានប៉ុន្មានថ្ងៃ?",
        "What is the password rotation policy?",
        "តើត្រូវធ្វើដូចម្តេចប្រសិនបើសង្ស័យអ៊ីមែលបន្លំ?",
    ],
)

if __name__ == "__main__":
    # 0.0.0.0 is required (not 127.0.0.1) so OpenShift's networking can
    # actually reach this pod from outside the container.
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
