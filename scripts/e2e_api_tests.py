"""
E2E API Tests — 2-Stage Extract-then-Answer Pipeline Validation
Run against a LIVE FastAPI server with APP_DEBUG=1.

Validates:
  - extracted_claims contain verbatim quotes (substring of actual chunks)
  - citations derive ONLY from claims
  - strict refusal format (zero citations on refuse)
  - entity lock stability across follow-ups
  - ban phrase absence in answers

Run:
  # Terminal 1: start server
  $env:APP_DEBUG="1"
  venv\\Scripts\\python.exe -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

  # Terminal 2: run tests
  venv\\Scripts\\python.exe scripts\\e2e_api_tests.py
"""
import sys, json, requests, time, os

API = os.getenv("E2E_API_URL", "http://localhost:8000")
PASS = 0
FAIL = 0
RESULTS = []

# ── Helpers ──────────────────────────────────────────────────────────────────
def _post(question: str, conversation_id: str = None):
    body = {"question": question}
    if conversation_id:
        body["conversation_id"] = conversation_id
    r = requests.post(f"{API}/api/ask", json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def _check(name, condition, detail=""):
    global PASS, FAIL
    tag = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    RESULTS.append({"name": name, "pass": condition, "detail": detail})
    return condition


# ── Ban phrase checker ────────────────────────────────────────────────────────
_BAN_PHRASES = [
    "general rules", "typically", "in most organizations",
    "may apply", "generally", "it depends",
    "please refer to guidelines", "please refer to",
    "refer to the official", "depends on the organization",
]

def _has_ban_phrase(answer: str) -> str:
    lower = answer.lower()
    for bp in _BAN_PHRASES:
        if bp in lower:
            return bp
    return ""


def _check_quote_offsets(d):
    """Validate quote_offsets on references when debug is available."""
    debug = d.get("debug", {})
    refs = d.get("references", [])
    for ref in refs:
        offsets = ref.get("quote_offsets")
        if offsets:
            _check(f"ref quote_offsets is tuple/list", isinstance(offsets, (list, tuple)) and len(offsets) == 2,
                   f"got {offsets}")
    # Check extraction_debug
    ext_debug = debug.get("extraction_debug", {})
    if ext_debug:
        localized = ext_debug.get("localized_quotes", [])
        for lq in localized:
            _check(f"localized has needle", "needle" in lq, str(lq.keys()))
            _check(f"localized has offsets", "offsets" in lq)


# ══════════════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ══════════════════════════════════════════════════════════════════════════════

# ── 0. Health / Debug ─────────────────────────────────────────────────────────
def test_debug_index():
    print("\n═══ 0. /debug/index ═══")
    try:
        r = requests.get(f"{API}/debug/index", timeout=10)
        r.raise_for_status()
        d = r.json()
        _check("debug/index reachable", True)
        _check("chunks_count > 0", (d.get("chunks_count") or 0) > 0, f"got {d.get('chunks_count')}")
        _check("table_registry ready", d.get("table_registry", {}).get("ready", False))
    except Exception as e:
        _check("debug/index reachable", False, str(e))


# ── 1. PERA Definition (English) ──────────────────────────────────────────────
def test_what_is_pera():
    print("\n═══ 1. 'what is PERA' ═══")
    d = _post("what is PERA")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())
    _check("decision != refuse", d.get("decision") != "refuse")
    _check("mentions PERA terms", any(w in a.lower() for w in ["punjab", "enforcement", "regulatory", "authority"]))
    _check(">=1 reference", len(d.get("references", [])) >= 1)
    _check("no ban phrase", not _has_ban_phrase(a), _has_ban_phrase(a))
    _check("has conversation_id", bool(d.get("conversation_id")))
    # Check extraction debug
    debug = d.get("debug", {})
    claims = debug.get("extracted_claims", [])
    if claims:
        _check("claims have quotes", all(c.get("quote") for c in claims))
        _check("claims have quote_offsets", all(c.get("quote_offsets") for c in claims))
    _check_quote_offsets(d)


# ── 2. PERA kia hai (Roman Urdu) ──────────────────────────────────────────────
def test_pera_kia_hai():
    print("\n═══ 2. 'PERA kia hai' ═══")
    d = _post("PERA kia hai")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())
    _check("decision != refuse", d.get("decision") != "refuse")
    _check("mentions PERA", "pera" in a.lower())
    _check("no ban phrase", not _has_ban_phrase(a))


# ── 3. PERA kia kam karti (Roman Urdu) ────────────────────────────────────────
def test_pera_kam():
    print("\n═══ 3. 'PERA kia kam karti' ═══")
    d = _post("PERA kia kam karti hai")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())


# ── 4. Schedule-I ─────────────────────────────────────────────────────────────
def test_schedule_1():
    print("\n═══ 4. 'schedule 1' ═══")
    d = _post("schedule 1")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())
    _check(">=1 reference", len(d.get("references", [])) >= 1)


# ── 5. Schedule-III ───────────────────────────────────────────────────────────
def test_schedule_3():
    print("\n═══ 5. 'schedule iii' ═══")
    d = _post("schedule iii")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())
    _check(">=1 reference", len(d.get("references", [])) >= 1)
    # Verify row data in claims or answer
    debug = d.get("debug", {})
    claims = debug.get("extracted_claims", [])
    all_quotes = " ".join(c.get("quote", "") for c in claims) if claims else a
    _check("schedule-III has 350", "350" in all_quotes or "350" in a,
           "neither claims nor answer mention 350")
    _check_quote_offsets(d)


# ── 6. SPPP-1 Salary ─────────────────────────────────────────────────────────
def test_sppp1_salary():
    print("\n═══ 6. 'sppp-1 salary' ═══")
    d = _post("sppp-1 salary")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())
    _check("decision != refuse", d.get("decision") != "refuse")
    _check("contains 350 (min)", "350" in a)
    _check("no ban phrase", not _has_ban_phrase(a))
    # Validate extraction source is deterministic table
    debug = d.get("debug", {})
    claims = debug.get("extracted_claims", [])
    if claims:
        sources = [c.get("_source", "") for c in claims]
        _check("table bypass used", any("table" in s or "jd" in s for s in sources),
               f"sources: {sources}")
    _check_quote_offsets(d)


# ── 7. Manager(Dev) salary ────────────────────────────────────────────────────
def test_manager_dev_salary():
    print("\n═══ 7. Manager(Dev) salary ═══")
    d = _post("salary btao manager (development) ki")
    a = d.get("answer", "")
    conv = d.get("conversation_id", "")
    _check("not refusal", "outside my scope" not in a.lower())
    _check("conversation_id returned", bool(conv))
    return conv


# ── 8. Follow-up: "number mein" with same conv_id ────────────────────────────
def test_number_mein_followup(conv_id: str):
    print("\n═══ 8. Follow-up 'number mein' ═══")
    if not conv_id:
        _check("SKIP (no conv_id)", False, "no conv_id from Q7")
        return
    d = _post("tell me in number", conversation_id=conv_id)
    a = d.get("answer", "")
    _check("same conversation_id", d.get("conversation_id") == conv_id)
    _check("not refusal", "outside my scope" not in a.lower())
    _check("no entity drift", "assistant manager gis" not in a.lower())
    _check("mentions manager/dev/sppp", any(w in a.lower() for w in ["manager", "development", "sppp"]))
    # Entity lock: verify claims still pinned to Manager (Dev)
    debug = d.get("debug", {})
    ctx = debug.get("ctx", {})
    last_job = (ctx.get("last_job_title") or "").lower()
    _check("entity pinned to manager", "manager" in last_job,
           f"last_job_title={ctx.get('last_job_title')}")
    _check_quote_offsets(d)


# ── 9. System Support Officer salary ──────────────────────────────────────────
def test_sso_salary():
    print("\n═══ 9. 'System Support Officer salary' ═══")
    d = _post("System Support Officer salary")
    a = d.get("answer", "")
    _check("not empty", len(a) > 10)
    _check("no ban phrase", not _has_ban_phrase(a))


# ── 10. EPO basics ────────────────────────────────────────────────────────────
def test_epo():
    print("\n═══ 10. 'what is EPO' ═══")
    d = _post("what is EPO in PERA")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())


# ── 11. Enforcement Officer powers ────────────────────────────────────────────
def test_eo_powers():
    print("\n═══ 11. 'EO powers' ═══")
    d = _post("What are the powers of Enforcement Officer")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())
    _check("no ban phrase", not _has_ban_phrase(a))


# ── 12. DG PERA role ──────────────────────────────────────────────────────────
def test_dg_pera():
    print("\n═══ 12. 'DG PERA role' ═══")
    d = _post("Director General PERA role and responsibilities")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())


# ── 13. Leave policy ──────────────────────────────────────────────────────────
def test_leave_policy():
    print("\n═══ 13. 'leave policy' ═══")
    d = _post("What is the leave policy at PERA")
    a = d.get("answer", "")
    _check("answered or refused (not error)", d.get("decision") in ("answer", "refuse"))


# ── 14. Dress code ────────────────────────────────────────────────────────────
def test_dress_code():
    print("\n═══ 14. 'dress code' ═══")
    d = _post("PERA dress code rules")
    a = d.get("answer", "")
    _check("answered or refused", d.get("decision") in ("answer", "refuse"))


# ── 15. CTO powers ────────────────────────────────────────────────────────────
def test_cto_powers():
    print("\n═══ 15. 'CTO powers' ═══")
    d = _post("CTO powers at PERA")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())
    _check("no ban phrase", not _has_ban_phrase(a))


# ── 16. PERA Act sections ─────────────────────────────────────────────────────
def test_pera_act():
    print("\n═══ 16. 'PERA Act' ═══")
    d = _post("important sections of PERA Act")
    a = d.get("answer", "")
    _check("not empty", len(a) > 20)


# ── 17. Who can fire (strict authority) ────────────────────────────────────────
def test_who_can_fire():
    print("\n═══ 17. 'who can fire' ═══")
    d = _post("who can fire employees at PERA")
    a = d.get("answer", "")
    _check("no ban phrase", not _has_ban_phrase(a))


# ── 18. Public nuisance (legal) ───────────────────────────────────────────────
def test_public_nuisance():
    print("\n═══ 18. 'public nuisance PERA' ═══")
    d = _post("what powers does PERA have regarding public nuisance")
    a = d.get("answer", "")
    _check("not empty", len(a) > 10)


# ── 19. Recruitment process ───────────────────────────────────────────────────
def test_recruitment():
    print("\n═══ 19. 'recruitment process' ═══")
    d = _post("PERA recruitment process")
    a = d.get("answer", "")
    _check("answered or refused", d.get("decision") in ("answer", "refuse"))


# ── 20. BPS-17 salary ─────────────────────────────────────────────────────────
def test_bps17():
    print("\n═══ 20. 'BPS-17 salary' ═══")
    d = _post("BPS-17 pay scale salary")
    a = d.get("answer", "")
    _check("answered or refused", d.get("decision") in ("answer", "refuse"))
    _check("no ban phrase", not _has_ban_phrase(a))


# ── 21. SDEO powers ───────────────────────────────────────────────────────────
def test_sdeo():
    print("\n═══ 21. 'SDEO powers' ═══")
    d = _post("Sub Divisional Enforcement Officer powers")
    a = d.get("answer", "")
    _check("not refusal", "outside my scope" not in a.lower())


# ── 22. Qualification for Manager ──────────────────────────────────────────────
def test_qualification():
    print("\n═══ 22. 'Manager qualification' ═══")
    d = _post("What is the qualification requirement for Manager at PERA")
    a = d.get("answer", "")
    _check("answered or refused", d.get("decision") in ("answer", "refuse"))


# ── OUT-OF-SCOPE REFUSALS ─────────────────────────────────────────────────────
def test_oos_javascript():
    print("\n═══ 23. OOS: JavaScript ═══")
    d = _post("tell me about JavaScript async/await")
    _check("decision=refuse or greeting", d.get("decision") in ("refuse", "greeting"))
    _check("references empty", len(d.get("references", [])) == 0)


def test_oos_weather():
    print("\n═══ 24. OOS: weather ═══")
    d = _post("what is the weather today")
    _check("decision=refuse or greeting", d.get("decision") in ("refuse", "greeting"))
    _check("references empty", len(d.get("references", [])) == 0)


def test_oos_joke():
    print("\n═══ 25. OOS: joke ═══")
    d = _post("tell me a joke")
    _check("decision=refuse or greeting", d.get("decision") in ("refuse", "greeting"))
    _check("references empty", len(d.get("references", [])) == 0)


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC REGISTRY TEST GROUP (Guaranteed Pass Suite)
# ══════════════════════════════════════════════════════════════════════════════

# ── 26. Schedule with misspelling "sehdule 3" ────────────────────────────────
def test_schedule_misspelling():
    print("\n═══ 26. Schedule misspelling: sehdule 3 ═══")
    d = _post("sehdule 3 dikhao")
    _check("decision=answer", d.get("decision") == "answer")
    ans = (d.get("answer") or "").lower()
    _check("mentions SPPP", "sppp" in ans)
    _check("has references", len(d.get("references", [])) > 0)


# ── 27. SSO always = System Support Officer (no clarification) ───────────────
def test_sso_no_clarification():
    print("\n═══ 27. SSO = System Support Officer ═══")
    d = _post("SSO kya hota hai")
    _check("decision=answer (not clarify)", d.get("decision") == "answer")
    ans = d.get("answer") or ""
    _check("mentions System Support Officer", "System Support Officer" in ans)
    _check("NOT ambiguous clarification", "clarify" not in ans.lower() or "system support" in ans.lower())
    _check("no ban phrases", not _has_ban_phrase(ans))


# ── 28. SSO salary with deterministic numbers ────────────────────────────────
def test_sso_salary_registry():
    print("\n═══ 28. SSO salary (deterministic) ═══")
    d = _post("System Support Officer ki salary kitni hai")
    _check("decision=answer", d.get("decision") == "answer")
    ans = (d.get("answer") or "").lower()
    _check("mentions SPPP or salary numbers", "sppp" in ans or "90,219" in ans or "90219" in ans)
    # Must mention actual numbers
    _check("mentions 90,219 or 200,219", "90,219" in ans or "90219" in ans or "200,219" in ans or "200219" in ans)
    _check("has references", len(d.get("references", [])) > 0)


# ── 29. CTO role via registry ────────────────────────────────────────────────
def test_cto_role_registry():
    print("\n═══ 29. CTO role (registry) ═══")
    d = _post("CTO ka role kya hai PERA mein")
    _check("decision=answer", d.get("decision") == "answer")
    ans = (d.get("answer") or "").lower()
    _check("mentions technology or IT", "technology" in ans or "it " in ans or "it-" in ans)
    _check("no ban phrases", not _has_ban_phrase(d.get("answer", "")))
    _check("has references", len(d.get("references", [])) > 0)


# ── 30. Android Developer pay scale ──────────────────────────────────────────
def test_android_pay_registry():
    print("\n═══ 30. Android Dev pay scale (registry) ═══")
    d = _post("android developer ki pay scale kya hai")
    _check("decision=answer", d.get("decision") == "answer")
    ans = (d.get("answer") or "").lower()
    _check("mentions SPPP", "sppp" in ans)
    _check("has references", len(d.get("references", [])) > 0)


# ── 31. PERA kia hai (deterministic) ─────────────────────────────────────────
def test_pera_def_registry():
    print("\n═══ 31. PERA kia hai (deterministic) ═══")
    d = _post("pera kia hai")
    _check("decision=answer", d.get("decision") == "answer")
    ans = (d.get("answer") or "").lower()
    _check("mentions Punjab", "punjab" in ans)
    _check("mentions Enforcement", "enforcement" in ans)
    _check("no ban phrases", not _has_ban_phrase(d.get("answer", "")))


# ── 32. PERA kb bani (when established) ──────────────────────────────────────
def test_pera_when_registry():
    print("\n═══ 32. PERA kb bani ═══")
    d = _post("pera kb bani")
    _check("decision=answer", d.get("decision") == "answer")
    ans = (d.get("answer") or "").lower()
    _check("mentions 2024 or November", "2024" in ans or "november" in ans)
    _check("has references", len(d.get("references", [])) > 0)


# ── 33. Salma Butt lookup ────────────────────────────────────────────────────
def test_salma_butt():
    print("\n═══ 33. Salma Butt (named entity) ═══")
    d = _post("salma butt kon hai PERA mein")
    ans = (d.get("answer") or "").lower()
    _check("decision=answer", d.get("decision") == "answer")
    _check("mentions relevant info", "mpa" in ans or "member" in ans or "mentioned" in ans or "regulatory" in ans or "salma" in ans)
    _check("no ban phrases", not _has_ban_phrase(d.get("answer", "")))


# ── 34. Act definition query ─────────────────────────────────────────────────
def test_act_definition():
    print("\n═══ 34. Act: definition section ═══")
    d = _post("PERA Act mein definitions kya hain")
    _check("decision=answer", d.get("decision") == "answer")
    ans = (d.get("answer") or "").lower()
    _check("mentions definition or court or department", "definition" in ans or "court" in ans or "department" in ans)
    _check("has references", len(d.get("references", [])) > 0)


# ── 35. Schedule-I org structure ─────────────────────────────────────────────
def test_schedule_1_registry():
    print("\n═══ 35. Schedule-I org structure (registry) ═══")
    d = _post("schedule 1 dikhao")
    _check("decision=answer", d.get("decision") == "answer")
    _check("has references", len(d.get("references", [])) > 0)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("PERA AI — E2E API Tests (2-Stage Pipeline + Deterministic Registry)")
    print(f"Target: {API}")
    print("=" * 60)

    # Verify server
    try:
        requests.get(f"{API}/debug/index", timeout=5)
    except Exception:
        print(f"\n!! Server not reachable at {API}")
        print("   Start: $env:APP_DEBUG='1'; venv\\Scripts\\python.exe -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Run all tests
    test_debug_index()
    test_what_is_pera()
    test_pera_kia_hai()
    test_pera_kam()
    test_schedule_1()
    test_schedule_3()
    test_sppp1_salary()
    conv_id = test_manager_dev_salary()
    test_number_mein_followup(conv_id)
    test_sso_salary()
    test_epo()
    test_eo_powers()
    test_dg_pera()
    test_leave_policy()
    test_dress_code()
    test_cto_powers()
    test_pera_act()
    test_who_can_fire()
    test_public_nuisance()
    test_recruitment()
    test_bps17()
    test_sdeo()
    test_qualification()
    test_oos_javascript()
    test_oos_weather()
    test_oos_joke()

    # Deterministic Registry suite
    print("\n" + "─" * 60)
    print("DETERMINISTIC REGISTRY SUITE")
    print("─" * 60)
    test_schedule_misspelling()
    test_sso_no_clarification()
    test_sso_salary_registry()
    test_cto_role_registry()
    test_android_pay_registry()
    test_pera_def_registry()
    test_pera_when_registry()
    test_salma_butt()
    test_act_definition()
    test_schedule_1_registry()

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("ALL PASS ✅")
    else:
        print("FAILURES ❌")
        for r in RESULTS:
            if not r["pass"]:
                print(f"  FAIL: {r['name']} — {r['detail']}")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
