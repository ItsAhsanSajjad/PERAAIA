"""
Production-Parity Probe — PERA AI Audit
Sends diagnostic queries to a running server and dumps full trace output.

Usage:
    APP_DEBUG=1 python -m uvicorn fastapi_app:app --port 8000
    python scripts/prod_parity_probe.py [--base http://127.0.0.1:8000]
"""
import argparse
import json
import time
import sys
import os
import requests

# ── Diagnostic query suite ────────────────────────────────────────────────────
# Each tuple: (label, question, expected_behaviour)
PROBE_SUITE = [
    # --- Greetings / out-of-scope (should NOT hit retrieval) ---
    ("greeting_en", "Hello, how are you?", "greeting response, no citations"),
    ("greeting_ur", "assalam o alaikum", "greeting response, no citations"),
    ("out_of_scope", "What's the weather in Lahore today?", "refuse, no citations"),

    # --- PERA Overview (semantic search) ---
    ("pera_what", "What is PERA?", "answer with 1+ citations"),
    ("pera_kya", "PERA kya hai?", "answer with 1+ citations (Urdu)"),

    # --- Schedule-I (table registry deterministic) ---
    ("schedule_1", "Show me Schedule-I offences and fines", "deterministic table hit"),

    # --- Schedule-III / SPPP (table registry deterministic) ---
    ("schedule_3", "What is the salary table in Schedule-III?", "deterministic table hit"),
    ("sppp_salary", "SPPP-1 salary kitni hai?", "table registry + salary figures"),

    # --- Job Description queries (registry + entity pinning) ---
    ("jd_cto", "What is the job description of CTO?", "JD for Chief Technology Officer"),
    ("jd_dg", "DG ki responsibilities kya hain?", "JD for Director General"),
    ("jd_sso", "SSO ka kya kaam hai?", "JD for System Support Officer (NOT Senior Staff Officer)"),

    # --- Act Section queries ---
    ("act_section_5", "What does Section 5 of the PERA Act say?", "Act section content"),
    ("act_section_21", "Section 21 PERA Act", "Act section content"),

    # --- Follow-up (session continuity) ---
    ("followup_salary", "uski salary kitni hai?", "should use entity from previous turn"),

    # --- Named person ---
    ("named_person", "Who is the current DG of PERA?", "named person or refusal"),

    # --- Edge: typo / misspelling ---
    ("typo_salary", "sallary of CTO", "handle typo -> salary of CTO"),
    ("typo_schedule", "sehdule 1", "handle typo -> Schedule-I"),
]


def run_probe(base_url: str, conv_id: str = None):
    """Run all probes and collect diagnostic trace data."""
    endpoint = f"{base_url}/api/ask"
    results = []
    history = []

    for label, question, expected in PROBE_SUITE:
        print(f"\n{'='*70}")
        print(f"[{label}] Q: {question}")
        print(f"  Expected: {expected}")
        print(f"{'='*70}")

        payload = {
            "question": question,
            "conversation_id": conv_id or "probe-session-001",
            "conversation_history": history[-6:],  # last 3 turns
        }

        try:
            t0 = time.time()
            resp = requests.post(endpoint, json=payload, timeout=60)
            elapsed = time.time() - t0

            if resp.status_code != 200:
                print(f"  ❌ HTTP {resp.status_code}: {resp.text[:200]}")
                results.append({
                    "label": label,
                    "question": question,
                    "expected": expected,
                    "status": resp.status_code,
                    "error": resp.text[:500],
                    "elapsed_s": round(elapsed, 2),
                })
                continue

            data = resp.json()
            answer = data.get("answer", "")
            refs = data.get("references", [])
            debug = data.get("debug", {})

            # Verdict
            has_answer = bool(answer and len(answer) > 20)
            has_refs = len(refs) > 0
            decision = debug.get("decision", data.get("decision", "?"))
            intent_c = debug.get("detected_intent_classify", "?")
            intent_r = debug.get("detected_intent_registry", "?")
            top_hits = debug.get("top_k_hits", [])[:5]

            verdict = "✅" if has_answer else "⚠️"
            if "refuse" in expected.lower() and decision == "refuse":
                verdict = "✅"
            elif "no citations" in expected.lower() and not has_refs:
                verdict = "✅"
            elif "citations" in expected.lower() and not has_refs and decision != "refuse":
                verdict = "❌ NO CITATIONS"

            print(f"  {verdict} Decision: {decision} | Intent(C): {intent_c} | Intent(R): {intent_r}")
            print(f"  Answer: {answer[:120]}...")
            print(f"  Refs: {len(refs)} | Hits: {debug.get('total_hits', '?')} | Time: {elapsed:.1f}s")
            if top_hits:
                print(f"  Top hit: score={top_hits[0].get('score')} tier={top_hits[0].get('tier')} doc={top_hits[0].get('doc', '?')[:40]}")

            # Check for SSO bug
            if label == "jd_sso":
                if "Senior Staff Officer" in answer:
                    print("  🐛 SSO BUG: Answered as 'Senior Staff Officer' instead of 'System Support Officer'")
                    verdict = "❌ SSO BUG"
                elif "System Support Officer" in answer:
                    print("  ✅ SSO correctly resolved to System Support Officer")

            results.append({
                "label": label,
                "question": question,
                "expected": expected,
                "verdict": verdict,
                "decision": decision,
                "intent_classify": intent_c,
                "intent_registry": intent_r,
                "registry_meta": debug.get("registry_meta", {}),
                "rewritten_query": debug.get("rewritten_query", ""),
                "answer_preview": answer[:200],
                "refs_count": len(refs),
                "refs": refs,
                "total_hits": debug.get("total_hits", 0),
                "top_hits": top_hits,
                "extracted_claims": debug.get("extracted_claims", []),
                "extraction_debug": debug.get("extraction_debug", {}),
                "elapsed_s": round(elapsed, 2),
            })

            # Maintain conversation history for follow-up test
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

        except requests.exceptions.ConnectionError:
            print(f"  ❌ Connection refused — is the server running at {base_url}?")
            sys.exit(1)
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append({
                "label": label,
                "question": question,
                "expected": expected,
                "error": str(e),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PROBE SUMMARY")
    print(f"{'='*70}")

    passes = sum(1 for r in results if r.get("verdict", "").startswith("✅"))
    fails = sum(1 for r in results if "❌" in r.get("verdict", ""))
    warnings = len(results) - passes - fails

    for r in results:
        v = r.get("verdict", "?")
        print(f"  {v:20s} [{r['label']:20s}] {r.get('decision', '?'):8s} | refs={r.get('refs_count', '?')} | {r.get('elapsed_s', '?')}s")

    print(f"\n  Passes: {passes}  Fails: {fails}  Warnings: {warnings}  Total: {len(results)}")

    # Save full trace
    out_path = os.path.join(os.path.dirname(__file__), "probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Full trace saved to: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PERA AI Production Parity Probe")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--conv-id", default=None, help="Conversation ID to use")
    args = parser.parse_args()

    run_probe(args.base, conv_id=args.conv_id)
