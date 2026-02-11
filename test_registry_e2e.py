"""Run just the deterministic registry E2E tests and write results to file."""
import sys, os
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

# Import test infrastructure
import requests, json

API = os.getenv("API_URL", "http://127.0.0.1:8000")
PASS = 0
FAIL = 0
RESULTS = []

def _post(question, conversation_id=None):
    payload = {"question": question, "conversation_history": []}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    r = requests.post(f"{API}/api/ask", json=payload, timeout=60)
    return r.json()

def _check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  [PASS] {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  [FAIL] {name} -- {detail}")

_BAN_PHRASES = [
    "general rules", "typically", "in most organizations",
    "may apply", "generally", "it depends",
    "please refer to guidelines", "please refer to",
    "refer to the official", "depends on the organization",
]

def _has_ban_phrase(answer):
    lower = answer.lower()
    return any(p in lower for p in _BAN_PHRASES)

# Tests
tests_run = []

def test_schedule_misspelling():
    tests_run.append("26. Schedule misspelling")
    d = _post("sehdule 3 dikhao")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = (d.get("answer") or "").lower()
    _check("mentions SPPP", "sppp" in ans)
    _check("has references", len(d.get("references", [])) > 0)

def test_sso_no_clarification():
    tests_run.append("27. SSO=System Support Officer")
    d = _post("SSO kya hota hai")
    _check("decision=answer (not clarify)", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = d.get("answer") or ""
    _check("mentions System Support Officer", "System Support Officer" in ans, f"answer: {ans[:100]}")
    _check("no ban phrases", not _has_ban_phrase(ans))

def test_sso_salary_registry():
    tests_run.append("28. SSO salary (deterministic)")
    d = _post("System Support Officer ki salary kitni hai")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = (d.get("answer") or "").lower()
    _check("mentions SPPP-5", "sppp-5" in ans or "sppp 5" in ans, f"answer: {ans[:100]}")
    _check("mentions numbers", "90,219" in ans or "90219" in ans or "200,219" in ans or "200219" in ans, f"answer: {ans[:100]}")
    _check("has references", len(d.get("references", [])) > 0, f"refs: {d.get('references', [])}")

def test_cto_role_registry():
    tests_run.append("29. CTO role (registry)")
    d = _post("CTO ka role kya hai PERA mein")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = (d.get("answer") or "").lower()
    _check("mentions technology or IT", "technology" in ans or "it " in ans or "it-" in ans, f"answer: {ans[:100]}")
    _check("has references", len(d.get("references", [])) > 0)

def test_android_pay_registry():
    tests_run.append("30. Android Dev pay scale")
    d = _post("android developer ki pay scale kya hai")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = (d.get("answer") or "").lower()
    _check("mentions SPPP", "sppp" in ans, f"answer: {ans[:100]}")
    _check("has references", len(d.get("references", [])) > 0)

def test_pera_def_registry():
    tests_run.append("31. PERA kia hai")
    d = _post("pera kia hai")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = (d.get("answer") or "").lower()
    _check("mentions Punjab", "punjab" in ans, f"answer: {ans[:100]}")
    _check("mentions Enforcement", "enforcement" in ans, f"answer: {ans[:100]}")

def test_pera_when_registry():
    tests_run.append("32. PERA kb bani")
    d = _post("pera kb bani")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = (d.get("answer") or "").lower()
    _check("mentions 2024 or November", "2024" in ans or "november" in ans, f"answer: {ans[:100]}")
    _check("has references", len(d.get("references", [])) > 0)

def test_salma_butt():
    tests_run.append("33. Salma Butt")
    d = _post("salma butt kon hai PERA mein")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = (d.get("answer") or "").lower()
    _check("mentions MPA or member", "mpa" in ans or "member" in ans, f"answer: {ans[:100]}")

def test_act_definition():
    tests_run.append("34. Act definitions")
    d = _post("PERA Act mein definitions kya hain")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    ans = (d.get("answer") or "").lower()
    _check("mentions definition/court/dept", "definition" in ans or "court" in ans or "department" in ans, f"answer: {ans[:100]}")
    _check("has references", len(d.get("references", [])) > 0)

def test_schedule_1_registry():
    tests_run.append("35. Schedule-I")
    d = _post("schedule 1 dikhao")
    _check("decision=answer", d.get("decision") == "answer", f"got {d.get('decision')}")
    _check("has references", len(d.get("references", [])) > 0)

# Run
try:
    requests.get(f"{API}/debug/index", timeout=5)
except Exception:
    with open("registry_e2e_results.txt", "w") as f:
        f.write("SERVER NOT REACHABLE\n")
    sys.exit(1)

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

# Write results
with open("registry_e2e_results.txt", "w", encoding="utf-8") as f:
    f.write("DETERMINISTIC REGISTRY E2E TESTS\n")
    f.write("=" * 50 + "\n\n")
    for t in tests_run:
        f.write(f"\n--- {t} ---\n")
    f.write("\n\nDETAILED RESULTS:\n")
    for r in RESULTS:
        f.write(r + "\n")
    f.write(f"\n\nSUMMARY: {PASS} passed, {FAIL} failed\n")
    if FAIL == 0:
        f.write("ALL PASS ✅\n")
    else:
        f.write("FAILURES ❌\n")
        for r in RESULTS:
            if "[FAIL]" in r:
                f.write(f"  {r}\n")

print(f"Done: {PASS} passed, {FAIL} failed")
print("Results written to registry_e2e_results.txt")
