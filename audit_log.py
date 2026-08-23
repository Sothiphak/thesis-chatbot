"""
audit_log.py -- tamper-evident audit logging for Control 5.

No real Red Hat AI Gateway component exists on this cluster (confirmed via
`oc get pods/deployment -n redhat-ods-applications | grep gateway` -- the
only "gateway" present is the RHOAI dashboard's own ingress route, unrelated
to model-access auditing). Standing up a full commercial API gateway product
was judged disproportionate to the time remaining, so this satisfies the
actual security property Control 5 requires -- tamper-evident audit trail of
every model request -- via a hash chain, at the application layer.

How it works: every log entry stores a hash of the PREVIOUS entry alongside
its own content. Altering or deleting any past entry breaks the chain for
every entry after it -- the same core mechanism blockchains use for tamper
evidence, without needing consensus, mining, or any of blockchain's actual
complexity. Genesis (first entry) chains to a well-known all-zero hash.

Stored on the Control-4 encrypted PVC (/data), so the audit log itself
inherits encryption at rest for free.
"""

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "/data/audit_log.jsonl")
GENESIS_HASH = "0" * 64

_lock = Lock()  # Gradio can handle concurrent requests; serialize log writes


def _canonical_json(d: dict) -> str:
    """Deterministic serialization -- sorted keys, so hash computation is
    reproducible regardless of dict insertion order."""
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _compute_entry_hash(entry_without_hash: dict) -> str:
    return hashlib.sha256(_canonical_json(entry_without_hash).encode("utf-8")).hexdigest()


def _get_last_entry_hash() -> str:
    path = Path(AUDIT_LOG_PATH)
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS_HASH
    with open(path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    if not lines:
        return GENESIS_HASH
    return json.loads(lines[-1])["entry_hash"]


def log_event(event_type: str, **fields) -> dict:
    """
    Append one tamper-evident entry. event_type examples used in app.py:
    'query_rejected' (relevance gate declined), 'query_answered' (reached
    vLLM successfully), 'query_error' (exception during generation).
    """
    with _lock:
        prev_hash = _get_last_entry_hash()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "prev_hash": prev_hash,
            **fields,
        }
        entry["entry_hash"] = _compute_entry_hash(entry)

        Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_canonical_json(entry) + "\n")

        return entry
