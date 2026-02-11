"""Quick test of deterministic_registry module - write to file."""
import sys, os
sys.path.insert(0, ".")
os.environ["APP_DEBUG"] = "0"

from deterministic_registry import route_intent, registry_answer

results = []

# Test intent routing
tests = [
    ("schedule 3", "SCHEDULE_LOOKUP"),
    ("sehdule III", "SCHEDULE_LOOKUP"),
    ("sched 1", "SCHEDULE_LOOKUP"),
    ("SSO ki salary", "JD_LOOKUP"),
    ("CTO ka role", "JD_LOOKUP"),
    ("android developer qualification", "JD_LOOKUP"),
    ("section 5", "ACT_SECTION"),
    ("pera kia hai", "PERA_DEFINITION"),
    ("pera kb bani", "PERA_DEFINITION"),
    ("salma butt kon hai", "NAMED_PERSON"),
    ("Manager Development ka role", "JD_LOOKUP"),
    ("hello", "FALLBACK"),
]
results.append("INTENT ROUTING TESTS:")
passes = 0
fails = 0
for q, expected in tests:
    intent, meta = route_intent(q)
    ok = intent == expected
    if ok:
        passes += 1
    else:
        fails += 1
    results.append(f"  [{'PASS' if ok else 'FAIL'}] '{q}' -> {intent} (expected {expected})")

results.append(f"\nRouting: {passes} pass, {fails} fail")

# Test registry answers
results.append("\nREGISTRY ANSWER TESTS:")
answer_tests = [
    ("schedule 3", True, "schedule_registry"),
    ("SSO ki salary", True, None),  # any source OK
    ("pera kia hai", True, None),
    ("pera kb bani", True, None),
    ("CTO ka role kya hai", True, None),
    ("android developer qualification", True, None),
    ("hello world", False, None),
]
a_passes = 0
a_fails = 0
for q, expect_result, expect_source in answer_tests:
    result = registry_answer(q)
    if expect_result:
        if result and result.get("claims"):
            src = result["debug"].get("source", "?")
            n = len(result["claims"])
            ok = True
            if expect_source:
                ok = src == expect_source
            if ok:
                a_passes += 1
            else:
                a_fails += 1
            results.append(f"  [{'PASS' if ok else 'FAIL'}] '{q}' -> {n} claims, source={src}")
        else:
            a_fails += 1
            results.append(f"  [FAIL] '{q}' -> no result (expected result)")
    else:
        if result is None:
            a_passes += 1
            results.append(f"  [PASS] '{q}' -> None (fallback)")
        else:
            a_fails += 1
            results.append(f"  [FAIL] '{q}' -> got result, expected None")

results.append(f"\nAnswers: {a_passes} pass, {a_fails} fail")
results.append(f"\nTOTAL: {passes + a_passes} pass, {fails + a_fails} fail")

# Write results
with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("Results written to test_results.txt")
print(f"TOTAL: {passes + a_passes} pass, {fails + a_fails} fail")
