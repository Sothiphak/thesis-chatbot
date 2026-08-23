"""
verify_audit_log.py -- checks the tamper-evident audit log's integrity.

Run this any time to prove the log hasn't been silently altered. Reports
exactly which entry the chain breaks at, if any -- not just pass/fail.

Usage: python3 verify_audit_log.py [path-to-audit-log.jsonl]
"""

import json
import hashlib
import sys
from pathlib import Path

GENESIS_HASH = "0" * 64


def canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def compute_entry_hash(entry_without_hash: dict) -> str:
    return hashlib.sha256(canonical_json(entry_without_hash).encode("utf-8")).hexdigest()


def verify_log(path: str) -> bool:
    text = Path(path).read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        print("Log is empty -- nothing to verify.")
        return True

    expected_prev = GENESIS_HASH
    all_valid = True

    for i, line in enumerate(lines):
        entry = json.loads(line)
        stored_hash = entry.get("entry_hash")
        stored_prev = entry.get("prev_hash")

        entry_copy = {k: v for k, v in entry.items() if k != "entry_hash"}
        recomputed_hash = compute_entry_hash(entry_copy)

        problems = []
        if stored_prev != expected_prev:
            problems.append(
                f"prev_hash mismatch: chain expected {expected_prev[:12]}..., "
                f"entry has {stored_prev[:12]}..."
            )
        if recomputed_hash != stored_hash:
            problems.append(
                f"content hash mismatch: recomputed {recomputed_hash[:12]}..., "
                f"stored {stored_hash[:12]}... (entry content was altered after logging)"
            )

        if problems:
            all_valid = False
            print(f"FAIL  entry {i} (timestamp={entry.get('timestamp')}): TAMPERING DETECTED")
            for p in problems:
                print(f"        - {p}")
        else:
            print(f"OK    entry {i}: {entry.get('event_type')}")

        # Chain forward using this entry's OWN stored hash (even if wrong) so
        # a single tampered entry is flagged precisely, rather than cascading
        # false "tampering" reports through every entry after it.
        expected_prev = stored_hash

    print()
    if all_valid:
        print(f"VERIFIED: all {len(lines)} entries pass. Chain intact, no tampering detected.")
    else:
        print("TAMPERING DETECTED: audit log integrity check FAILED.")
    return all_valid


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/data/audit_log.jsonl"
    result = verify_log(path)
    sys.exit(0 if result else 1)
