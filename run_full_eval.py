"""Comprehensive evaluation of all 54 test cases for PERA AI Assistant."""
import time, json, sys
from retriever import retrieve
from answerer import answer_question

# ── Test Case definitions ──────────────────────────────────────────────
# Each: (id, query, must_contain_keywords, is_refusal, history_turns)
# For refusal tests, must_contain_keywords is empty and is_refusal=True
# For multi-turn, history_turns is a list of (user, assistant_stub) pairs

TESTS = [
    # A) Scope, Title, Commencement
    ("TC-001", "What is the short title of the Act?",
     ["punjab enforcement", "2024"], False, None),
    ("TC-002", "To which area does this Act extend?",
     ["whole of the punjab"], False, None),
    ("TC-003", "When does this Act come into force?",
     ["gazette", "notify"], False, None),

    # B) Definitions
    ("TC-004", "Define 'Authority' under the Act.",
     ["authority"], False, None),
    ("TC-005", "Who is an 'Authorized Officer'?",
     ["authorized officer"], False, None),
    ("TC-006", "What are 'Scheduled laws'?",
     ["schedule"], False, None),
    ("TC-007", "Define 'public nuisance'.",
     ["nuisance"], False, None),
    ("TC-008", "What does 'State property' mean?",
     ["state property"], False, None),

    # C) Establishment + Structure
    ("TC-009", "How is the Authority established?",
     ["notification", "gazette"], False, None),
    ("TC-010", "Is the Authority a legal corporate body?",
     ["body corporate"], False, None),
    ("TC-011", "Where is the headquarters of the Authority?",
     ["lahore"], False, None),
    ("TC-012", "Who is the Chairperson of the Authority?",
     ["chief minister"], False, None),
    ("TC-013", "Can the Chairperson delegate authority?",
     ["delegate", "vice chairperson"], False, None),

    # D) Powers & Functions
    ("TC-014", "What are the overall powers of the Authority?",
     ["power"], False, None),
    ("TC-015", "Who appoints Hearing Officers and Authorized Officers?",
     ["authority", "appoint"], False, None),
    ("TC-016", "Can the Authority delegate its powers?",
     ["delegate"], False, None),

    # E) District Board
    ("TC-017", "How is the District Board formed and who chairs it?",
     ["deputy commissioner"], False, None),
    ("TC-018", "Where is the Board housed?",
     ["district", "headquarter"], False, None),
    ("TC-019", "What are the Board's main functions?",
     ["enforcement"], False, None),

    # F) Hearing Officer
    ("TC-020", "Who can file a representation to the Hearing Officer and in what matters?",
     ["hearing officer"], False, None),
    ("TC-021", "Can the Hearing Officer issue an Absolute Order?",
     ["absolute order"], False, None),
    ("TC-022", "Are hearings under this Act formal court trials?",
     ["summari"], False, None),

    # G) Enforcement Stations
    ("TC-023", "Who establishes an Enforcement Station?",
     ["government", "notification"], False, None),
    ("TC-024", "What staff positions exist in an Enforcement Station?",
     ["sdeo"], False, None),
    ("TC-025", "Who is responsible for record keeping at the station?",
     ["sub divisional enforcement officer"], False, None),

    # H) SDEO Powers
    ("TC-026", "Can the SDEO register FIRs and investigate?",
     ["fir"], False, None),
    ("TC-027", "Can the SDEO make arrests?",
     ["arrest"], False, None),
    ("TC-028", "Can the SDEO issue summons/notices?",
     ["summon", "notice"], False, None),

    # I) Enforcement Officer Powers
    ("TC-029", "Can an Enforcement Officer enter and inspect public property without a warrant?",
     ["public property"], False, None),
    ("TC-030", "Can an Enforcement Officer search private premises?",
     ["warrant", "magistrate"], False, None),
    ("TC-031", "Can an Enforcement Officer stop and search a vehicle/person?",
     ["vehicle", "search"], False, None),
    ("TC-032", "Can the Enforcement Officer seize items and must they give a receipt?",
     ["receipt"], False, None),
    ("TC-033", "Can an Enforcement Officer impose fines?",
     ["fine"], False, None),
    ("TC-034", "Can an Enforcement Officer issue an EPO?",
     ["epo"], False, None),

    # J) Investigation Officer Powers
    ("TC-035", "Can an Investigation Officer confiscate a vehicle?",
     ["confiscat"], False, None),
    ("TC-036", "Can an Investigation Officer arrest an offender?",
     ["arrest"], False, None),

    # K) Arrest Rules
    ("TC-037", "When can PERA staff exercise power to arrest?",
     ["arrest"], False, None),
    ("TC-038", "After arrest, what must happen next?",
     ["24 hour", "magistrate"], False, None),

    # L) Public Nuisance
    ("TC-039", "Who can file a complaint about a public nuisance and to whom?",
     ["complaint", "nuisance"], False, None),
    ("TC-040", "What happens if the Enforcement Officer finds a public nuisance?",
     ["epo"], False, None),
    ("TC-041", "If the offender does not comply with the EPO, what actions can be taken?",
     ["fine"], False, None),
    ("TC-042", "Can the Hearing Officer convert an EPO into an Absolute Order?",
     ["absolute order"], False, None),

    # M) Encroachments
    ("TC-043", "Is encroachment on State property prohibited?",
     ["prohibit", "encroach"], False, None),
    ("TC-044", "What is the procedure for removal of movable encroachment?",
     ["remov", "encroach"], False, None),
    ("TC-045", "What is the procedure for removal of immovable encroachment?",
     ["hearing officer", "encroach"], False, None),

    # N) Notices
    ("TC-046", "How can a notice be served under the Act?",
     ["served", "notice"], False, None),

    # O) Penalties + Offences
    ("TC-047", "Can PERA impose any penalty it wants under this Act?",
     ["scheduled law"], False, None),
    ("TC-048", "What happens if someone disobeys an order or obstructs officers?",
     ["fine", "offence"], False, None),
    ("TC-049", "If a company commits an offence, who is liable?",
     ["director", "company"], False, None),

    # P) Refusal/Guardrail Tests
    ("TC-050", "What is the salary of an Enforcement Officer?",
     [], True, None),
    ("TC-051", "Which petrol stations are illegal in Lahore right now?",
     [], True, None),
    ("TC-052", "Give me the list of all Scheduled laws and their sections.",
     ["schedule"], False, None),  # Partial answer OK if schedule is in context
]

# Multi-turn tests
MULTI_TURN_TESTS = [
    ("TC-053", 
     [("Can an Enforcement Officer search private property?", None),
      ("Does he need a warrant for that?", None)],
     ["warrant", "magistrate"]),
    ("TC-054",
     [("What happens after arrest?", None),
      ("Within how many hours must the person be produced before the Magistrate?", None)],
     ["24"]),
]

def check_pass(answer: str, keywords: list, is_refusal: bool, decision: str) -> tuple:
    """Check if answer passes. Returns (passed, reason)."""
    lower = answer.lower()
    
    if is_refusal:
        # Must refuse or contain refusal markers
        refusal_markers = ["available nahi hai", "pera se contact", "not available",
                          "not found", "not contain", "nahi hai", "not specified",
                          "unfortunately", "i could not find"]
        if decision == "refuse" or any(m in lower for m in refusal_markers):
            return True, "Correctly refused"
        return False, f"Should have refused but answered: {answer[:100]}"
    
    # Must answer (not refuse)
    refusal_markers = ["available nahi hai", "pera se contact karein"]
    if any(m in lower for m in refusal_markers):
        return False, f"Refused when should have answered"
    
    # Check keywords
    missing = [k for k in keywords if k.lower() not in lower]
    if missing:
        return False, f"Missing keywords: {missing}"
    
    return True, "All keywords found"


def run_single(tc_id, query, keywords, is_refusal, history=None):
    """Run a single test case."""
    try:
        ret = retrieve(query)
        res = answer_question(query, ret, conversation_history=history)
        answer = res.get("answer", "")
        decision = res.get("decision", "")
        passed, reason = check_pass(answer, keywords, is_refusal, decision)
        return {
            "id": tc_id, "query": query, "passed": passed,
            "reason": reason, "answer": answer[:200], "decision": decision
        }
    except Exception as e:
        return {
            "id": tc_id, "query": query, "passed": False,
            "reason": f"ERROR: {e}", "answer": "", "decision": "error"
        }


def main():
    results = []
    total = len(TESTS) + len(MULTI_TURN_TESTS)
    
    print(f"Running {total} test cases...\n")
    
    # Single-turn tests
    for i, (tc_id, query, keywords, is_refusal, history) in enumerate(TESTS):
        print(f"  [{i+1}/{total}] {tc_id}: {query[:50]}...", end=" ", flush=True)
        r = run_single(tc_id, query, keywords, is_refusal, history)
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(status)
        results.append(r)
    
    # Multi-turn tests
    for mti, (tc_id, turns, final_keywords) in enumerate(MULTI_TURN_TESTS):
        idx = len(TESTS) + mti + 1
        print(f"  [{idx}/{total}] {tc_id} (multi-turn)...", end=" ", flush=True)
        
        history = []
        for turn_q, _ in turns[:-1]:
            # Execute previous turns to build history
            ret = retrieve(turn_q)
            res = answer_question(turn_q, ret, conversation_history=history if history else None)
            history.append({"role": "user", "content": turn_q})
            history.append({"role": "assistant", "content": res.get("answer", "")})
        
        # Final turn with history
        final_q = turns[-1][0]
        r = run_single(tc_id, final_q, final_keywords, False, history)
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(status)
        results.append(r)
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    
    # Write report
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"PERA AI EVALUATION REPORT")
    lines.append(f"{'='*80}")
    lines.append(f"Total: {total} | Passed: {passed} | Failed: {failed} | Rate: {passed/total*100:.0f}%")
    lines.append(f"{'='*80}\n")
    
    # Failures first
    failures = [r for r in results if not r["passed"]]
    if failures:
        lines.append("FAILURES:")
        lines.append("-"*40)
        for r in failures:
            lines.append(f"{r['id']}: {r['query'][:60]}")
            lines.append(f"  Reason: {r['reason']}")
            lines.append(f"  Answer: {r['answer'][:150]}")
            lines.append("")
    
    lines.append("\nALL RESULTS:")
    lines.append("-"*40)
    for r in results:
        mark = "✅" if r["passed"] else "❌"
        lines.append(f"{mark} {r['id']}: {r['reason'][:80]}")
    
    report = "\n".join(lines)
    with open("eval_report_full.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")
    print(f"Full report: eval_report_full.txt")


if __name__ == "__main__":
    main()
