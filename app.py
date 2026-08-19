"""
Sovereign Khmer RAG Chatbot — with model selection and improved UI.

Lets the user pick between the base model and the fine-tuned model,
side by side, while keeping the same reliable RAG grounding for both.

Built using gr.Blocks() directly rather than the higher-level
gr.ChatInterface(), since we already hit a version-compatibility issue
with ChatInterface's constructor once tonight — Blocks is the more
stable, lower-level API, less likely to surprise us again.
"""

import os
import gradio as gr
import httpx
from openai import OpenAI
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

DB_PATH = "milvus_rag.db"
COLLECTION_NAME = "hr_it_policies"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TOP_K = 3

VLLM_INTERNAL_URL = os.environ.get(
    "VLLM_URL", "http://vllm-sealion.model-serving.svc.cluster.local:8000/v1"
)

# Scoped CA trust for the vLLM connection specifically -- deliberately NOT
# using SSL_CERT_FILE/REQUESTS_CA_BUNDLE process-wide env vars, since those
# REPLACE Python's default trust store rather than extend it, which broke
# unrelated HuggingFace Hub downloads (the embedding model fetch) the first
# time this was attempted. This httpx.Client is scoped to vllm_client only --
# every other HTTPS call in this process (HuggingFace Hub included) keeps
# using the normal system trust store, completely untouched.
VLLM_CA_BUNDLE = os.environ.get("VLLM_CA_BUNDLE")

MODEL_DISPLAY_MAP = {
    "Standard": "sealion",
    "Enhanced (Fine-tuned)": "sealion-ft",
}

# --- Relevance gate ---
# Fixes a real, confirmed bug: without this, an off-topic question (e.g. "what's
# the best Khmer food?") still runs the full RAG pipeline, retrieves the
# nearest-but-irrelevant HR/IT chunks anyway, and the model fabricates an
# answer while citing those unrelated documents as its "source." Gating on
# the top retrieval score catches this BEFORE generation -- so it's also
# faster for off-topic queries, not just more correct, since no LLM call
# happens at all on the fast-reject path.
#
# Calibrated 2026-08-19 against the real 15-document collection (see
# calibrate_relevance.py output): in-scope queries scored 0.8238-0.8765,
# out-of-scope scored 0.7344-0.8096. Set slightly below the in-scope floor
# rather than the exact midpoint (0.8167) -- deliberately biased toward
# rejecting borderline cases rather than risking a hallucinated answer,
# consistent with this thesis's core no-hallucination argument.
RELEVANCE_THRESHOLD = 0.82

OUT_OF_SCOPE_MESSAGE = (
    "សូមអភ័យទោស ខ្ញុំមិនមានព័ត៌មានទាក់ទងនឹងសំណួរនេះទេ។ "
    "ខ្ញុំអាចឆ្លើយបានតែសំណួរទាក់ទងនឹងគោលការណ៍ HR/IT របស់ក្រុមហ៊ុនប៉ុណ្ណោះ។\n\n"
    "Sorry, I don't have information related to that question. "
    "I can only answer questions about company HR/IT policies."
)

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Connecting to knowledge base at {DB_PATH}...")
milvus_client = MilvusClient(DB_PATH)
milvus_client.load_collection(COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' loaded and ready.")

# Sanity check: the relevance gate below assumes COSINE or IP metric, where a
# HIGHER distance score means MORE similar. If this collection actually uses
# L2, that assumption is backwards and the gate will do the opposite of what
# it's supposed to -- printed once at startup so it's impossible to miss.
try:
    _collection_info = milvus_client.describe_collection(COLLECTION_NAME)
    print(f"Collection metric info (verify this is COSINE or IP, not L2): {_collection_info}")
except Exception as _e:
    print(f"Could not verify collection metric type automatically: {_e}")
    print("Manually confirm via your ingest script before trusting RELEVANCE_THRESHOLD.")

print(f"Connecting to vLLM at {VLLM_INTERNAL_URL}...")
if VLLM_INTERNAL_URL.startswith("https://") and VLLM_CA_BUNDLE:
    print(f"Using scoped CA bundle for vLLM TLS verification: {VLLM_CA_BUNDLE}")
    # Explicitly matching the OpenAI SDK's own generous default (read=600s,
    # sized for slow LLM generation) rather than leaving the underlying
    # httpx.Client on its bare 5s default. NOTE: a direct reproduction test
    # found the bare 5s default did NOT actually cause a failure against a
    # 7s-delayed response -- so this is a safe, sensible alignment with the
    # SDK's own intended behavior, but NOT a confirmed fix for the specific
    # "Connection error" seen in production. The real cause of that is still
    # unknown -- see the improved exception logging below, which is the
    # actual next diagnostic step.
    vllm_http_client = httpx.Client(
        verify=VLLM_CA_BUNDLE,
        timeout=httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=600.0),
    )
    vllm_client = OpenAI(base_url=VLLM_INTERNAL_URL, api_key="not-needed", http_client=vllm_http_client)
else:
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


def extract_text(message) -> str:
    """
    Normalize a Gradio chat message to plain text.

    Found via real debug logs: Gradio's chatbot history 'content' field was
    observed coming through as [{'text': '...', 'type': 'text'}] -- a list
    of content-part dicts -- rather than a plain string, for ordinary text
    input. This silently corrupted retrieval (the embedding model encoded
    literal Python list/dict syntax as part of the query, measurably
    shifting relevance scores vs. the clean text used in calibration) and
    very likely broke the actual vLLM call outright: the OpenAI client
    validates `content` client-side before ever sending a request, so a
    malformed value here can throw a *local* exception that the code was
    mislabeling as a connection failure.
    """
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return " ".join(
            part.get("text", "")
            for part in message
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(message)


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


def get_response(message: str, history: list, model_choice: str) -> str:
    model_id = MODEL_DISPLAY_MAP.get(model_choice, "sealion")

    retrieved = retrieve(message)

    # Relevance gate: check BEFORE building any prompt or calling the LLM.
    # An empty retrieval result, or a top score below threshold, means this
    # question likely isn't answerable from the knowledge base -- reject
    # fast and honestly instead of letting the model improvise.
    top_score = retrieved[0]["distance"] if retrieved else 0.0
    print(f"[relevance] query={message[:60]!r} top_score={top_score:.4f} threshold={RELEVANCE_THRESHOLD}")

    if not retrieved or top_score < RELEVANCE_THRESHOLD:
        return OUT_OF_SCOPE_MESSAGE

    system_prompt, sources = build_augmented_prompt(message, retrieved)

    # history is already a list of {"role": ..., "content": ...} dicts
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    try:
        response = vllm_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.3,
            max_tokens=400,
            extra_body={"repetition_penalty": 1.15},
        )
        answer = response.choices[0].message.content
        source_note = f"\n\n📄 *Retrieved from: {', '.join(sources)}*"
        return answer + source_note
    except Exception as e:
        # Log the REAL exception to server logs -- the generic message
        # shown to the user was masking the actual cause (a client-side
        # validation error, in the incident this comment refers to, not
        # an actual network/connection failure) and made this bug take
        # much longer to diagnose than it should have.
        import traceback
        print(f"[get_response] EXCEPTION type={type(e).__name__} message={e}")
        traceback.print_exc()
        return f"⚠️ Could not reach the model server. Error: {e}"


CUSTOM_CSS = """
.gradio-container {
    font-family: 'Noto Sans Khmer', 'Inter', sans-serif !important;
    max-width: 900px !important;
    margin: auto !important;
}
#header-md h1 {
    background: linear-gradient(90deg, #0d9488, #0891b2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 0.2em;
}
#header-md p {
    color: #64748b;
    font-size: 0.95em;
}
.model-radio label {
    font-weight: 600;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="Sovereign Khmer RAG Chatbot",
) as demo:

    gr.Markdown(
        """
        # 🇰🇭 Sovereign Khmer RAG Chatbot
        Internal employee assistant, grounded in real company policy documents, running entirely on
        institution-controlled infrastructure. Ask in Khmer or English.
        """,
        elem_id="header-md",
    )

    model_choice = gr.Radio(
        choices=["Standard", "Enhanced (Fine-tuned)"],
        value="Standard",
        label="🧠 Model",
        info="Standard = base model  •  Enhanced = fine-tuned for improved Khmer fluency",
        elem_classes="model-radio",
    )

    chatbot = gr.Chatbot(height=460)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="សរសេរសំណួររបស់អ្នកនៅទីនេះ... / Type your question here...",
            show_label=False,
            scale=8,
            container=False,
        )
        send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=100)

    clear_btn = gr.Button("🗑️ Clear conversation", size="sm")

    gr.Examples(
        examples=[
            "តើថ្ងៃឈប់សម្រាកប្រចាំឆ្នាំមានប៉ុន្មានថ្ងៃ?",
            "What is the password rotation policy?",
            "តើត្រូវធ្វើដូចម្តេចប្រសិនបើសង្ស័យអ៊ីមែលបន្លំ?",
            "How many days of paid sick leave do I get?",
        ],
        inputs=msg,
        label="Try one of these:",
    )

    def user_submit(message, history):
        if not message.strip():
            return "", history
        history = history + [{"role": "user", "content": message}]
        return "", history

    def bot_respond(history, model_choice):
        if not history or history[-1]["role"] != "user":
            return history
        user_message = extract_text(history[-1]["content"])
        response = get_response(user_message, history[:-1], model_choice)
        history = history + [{"role": "assistant", "content": response}]
        return history

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, [chatbot, model_choice], chatbot
    )
    send_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, [chatbot, model_choice], chatbot
    )
    clear_btn.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"),
        css=CUSTOM_CSS,
    )
