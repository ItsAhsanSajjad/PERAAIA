"""
Coverage Lock E2E Test Suite — test_tc_suite_min.py
Validates known-in-doc queries return deterministic answers with correct metadata.
Run: python scripts/test_tc_suite_min.py --base http://127.0.0.1:8000
"""

import requests, json, sys, time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:8000")
args = parser.parse_args()
BASE = args.base.rstrip("/")

BANNED_PHRASES = [
    "outside my scope",
    "I cannot",
    "I don't have",
    "not available in",
    "I'm unable",
]

# ═══════════════════════════════════════════════════════════════════
#  Test cases: (label, query, expected_intent, must_answer, must_contain, banned)
# ═══════════════════════════════════════════════════════════════════
CASES = [
    # Schedule typos + Roman Urdu
    ("sched1_normal",   "schedule 1 kia hai?",       "SCHEDULE_LOOKUP",    True,  ["Schedule-I"],          []),
    ("sched1_typo1",    "sehdule 1 dikhao",          "SCHEDULE_LOOKUP",    True,  ["Schedule-I"],          []),
    ("sched1_typo2",    "schudule 1 batao",          "SCHEDULE_LOOKUP",    True,  ["Schedule-I"],          []),
    ("sched2_normal",   "schedule 2 references",     "SCHEDULE_LOOKUP",    True,  ["Schedule-II"],         []),
    ("sched2_typo",     "shedule 2 kia hy",          "SCHEDULE_LOOKUP",    True,  [],                      []),
    ("sched3_normal",   "schedule 3 pay table",      "SCHEDULE_LOOKUP",    True,  ["SPPP"],                []),
    ("sched3_typo",     "schdule 3 salary",          "SCHEDULE_LOOKUP",    True,  ["SPPP"],                []),

    # SSO role
    ("sso_role",        "SSO ka role kia hai?",      "JD_LOOKUP",          True,  ["System Support"],      []),
    ("sso_salary",      "SSO ki salary kitni hai?",  "JD_LOOKUP",          True,  ["SPPP-5"],              []),

    # CTO powers (synonym of responsibilities)
    ("cto_powers",      "CTO ke powers kya hain?",   "JD_LOOKUP",          True,  [],                      BANNED_PHRASES),
    ("cto_role",        "CTO kia karta hai?",        "JD_LOOKUP",          True,  [],                      BANNED_PHRASES),

    # Android pay scale
    ("android_salary",  "Android Developer salary?", "JD_LOOKUP",          True,  ["SPPP-4"],              []),

    # PERA definition
    ("pera_what",       "PERA kya hai?",             "PERA_DEFINITION",    True,  ["Punjab Enforcement"],  []),
    ("pera_kia",        "PERA kia hy?",              "PERA_DEFINITION",    True,  ["Punjab Enforcement"],  []),
    ("pera_what_en",    "What is PERA?",             "PERA_DEFINITION",    True,  ["Punjab Enforcement"],  []),

    # PERA commencement — must NOT invent a date
    ("pera_kb_bani",    "PERA kb bani?",             "PERA_COMMENCEMENT",  True,  [],                      ["notification"]),
    ("pera_when",       "When was PERA established?","PERA_COMMENCEMENT",  True,  [],                      []),

    # DG reporting
    ("dg_report",       "DG kisko reporting hy?",    "NAMED_PERSON",       True,  [],                      []),

    # DG name — strict: answer ONLY if name found in chunks
    ("dg_name",         "DG ka nam kia hy?",         "NAMED_PERSON",       None,  [],                      []),  # None = we don't know if name is in chunks

    # Act definitions
    ("act_mean",        "act mean in authority?",     "ACT_DEFINITION",     True,  [],                      BANNED_PHRASES),
    ("authority_mean",  "authority ka matlab?",       "ACT_DEFINITION",     True,  [],                      BANNED_PHRASES),

    # Section 10 powers
    ("sec10_powers",    "Section 10 ki powers?",     "ACT_SECTION",        True,  [],                      BANNED_PHRASES),
]


def send_query(q: str) -> dict:
    """Send a query to the API and return the full JSON response."""
    try:
        r = requests.post(
            f"{BASE}/api/ask",
            json={"question": q, "user_id": "tc_suite", "message": q, "conversation_history": []},
            timeout=60,
        )
        if r.status_code != 200:
            return {"_error": f"HTTP {r.status_code}", "_body": r.text[:200]}
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def run_tests():
    results = []
    passes = 0
    fails = 0
    total = len(CASES)

    for label, query, expected_intent, must_answer, must_contain, banned in CASES:
        print(f"\n{'='*60}")
        print(f"  [{label}] {query}")
        print(f"  Expected intent: {expected_intent}")

        resp = send_query(query)

        if "_error" in resp:
            print(f"  ❌ ERROR: {resp['_error']}")
            results.append({"label": label, "verdict": "❌", "error": resp["_error"]})
            fails += 1
            continue

        answer = resp.get("answer", "")
        decision = resp.get("decision", "?")
        refs = resp.get("references", [])
        debug = resp.get("debug", {})
        det_intent = debug.get("detected_intent_registry", "?")
        det_meta = debug.get("registry_meta", {})
        claims = debug.get("extracted_claims", [])

        # Check intent routing
        intent_ok = det_intent == expected_intent
        if not intent_ok:
            print(f"  ⚠️  Intent: {det_intent} (expected {expected_intent})")

        # Check decision
        decision_ok = True
        if must_answer is True and decision == "refuse":
            decision_ok = False
            print(f"  ⚠️  Decision: REFUSE (expected ANSWER)")
        elif must_answer is False and decision != "refuse":
            decision_ok = False
            print(f"  ⚠️  Decision: {decision} (expected REFUSE)")

        # Check citations page >= 1
        pages_ok = True
        for ref in refs:
            ps = ref.get("page_start", 0)
            if isinstance(ps, int) and ps < 1:
                pages_ok = False
                print(f"  ⚠️  page_start={ps} in ref")

        # Check must_contain
        contain_ok = True
        for term in must_contain:
            found_in_answer = term.lower() in answer.lower()
            found_in_claims = any(term.lower() in str(c).lower() for c in claims)
            if not found_in_answer and not found_in_claims:
                contain_ok = False
                print(f"  ⚠️  Missing required term: '{term}'")

        # Check banned phrases
        banned_ok = True
        for phrase in banned:
            if phrase.lower() in answer.lower():
                banned_ok = False
                print(f"  ⚠️  Contains banned phrase: '{phrase}'")

        # Overall verdict
        all_ok = intent_ok and decision_ok and pages_ok and contain_ok and banned_ok
        # For must_answer=None (DG name), skip decision check
        if must_answer is None:
            all_ok = intent_ok and pages_ok and contain_ok and banned_ok

        verdict = "✅" if all_ok else "❌"
        if all_ok:
            passes += 1
        else:
            fails += 1

        print(f"  {verdict} intent={det_intent} decision={decision} refs={len(refs)} "
              f"claims={len(claims)} pages_ok={pages_ok}")
        print(f"  Answer: {answer[:120]}...")

        result = {
            "label": label,
            "query": query,
            "verdict": verdict,
            "expected_intent": expected_intent,
            "actual_intent": det_intent,
            "decision": decision,
            "refs_count": len(refs),
            "claims_count": len(claims),
            "pages_ok": pages_ok,
            "contain_ok": contain_ok,
            "banned_ok": banned_ok,
            "answer_preview": answer[:200],
            "refs": refs[:3],
        }
        results.append(result)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {passes}/{total} passed, {fails}/{total} failed")
    print(f"{'='*60}")

    # Save results
    out_path = "scripts/tc_suite_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Results saved to {out_path}")

    return passes == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
