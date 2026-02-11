"""
Unit tests for locate_quote() and normalize_with_map() — deterministic localization.

Tests cover:
  - Clean text (exact match)
  - Soft hyphens (\\u00AD) embedded in words
  - Line-break hyphenation  (enforce-\\nment → enforcement)
  - Random extra whitespace / tabs / newlines
  - Unicode NFKC normalization (ligatures, fullwidth chars)
  - Table-formatted text with pipes/dashes
  - No-match case → None
  - Offset correctness: chunk_text[start:end] == quote

Run:
  venv\\Scripts\\python.exe scripts\\test_quote_localization.py
"""
import sys, os, random, string

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answerer import normalize_with_map, locate_quote, extract_table_row

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
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
    return condition


# ===========================================================================
# normalize_with_map tests
# ===========================================================================
def test_normalize_basic():
    print("\n═══ normalize_with_map: basic ═══")
    text = "Hello  World"
    norm, n2o = normalize_with_map(text)
    check("lowercase", norm == "hello world")
    check("map length matches norm", len(n2o) == len(norm))
    # Verify mapping: 'h' at original index 0
    check("n2o[0] -> 0", n2o[0] == 0)


def test_normalize_soft_hyphen():
    print("\n═══ normalize_with_map: soft hyphen ═══")
    text = "en\u00ADforce\u00ADment"
    norm, n2o = normalize_with_map(text)
    check("soft hyphens removed", norm == "enforcement")
    check("map length matches", len(n2o) == len(norm))


def test_normalize_line_break_hyphen():
    print("\n═══ normalize_with_map: line-break hyphen ═══")
    text = "enforce-\nment of law"
    norm, n2o = normalize_with_map(text)
    check("de-hyphenated", "enforcement" in norm, f"got: {norm}")
    check("rest preserved", "of law" in norm)


def test_normalize_crlf_hyphen():
    print("\n═══ normalize_with_map: CRLF hyphen ═══")
    text = "enforce-\r\nment"
    norm, n2o = normalize_with_map(text)
    check("de-hyphenated CRLF", "enforcement" in norm, f"got: {norm}")


def test_normalize_whitespace_collapse():
    print("\n═══ normalize_with_map: whitespace collapse ═══")
    text = "the   quick\t \nbrown   fox"
    norm, n2o = normalize_with_map(text)
    check("whitespace collapsed", norm == "the quick brown fox")


def test_normalize_nfkc():
    print("\n═══ normalize_with_map: NFKC ═══")
    # ﬁ (U+FB01) should normalize to 'fi'
    text = "of\ufb01cial"
    norm, n2o = normalize_with_map(text)
    check("ligature normalized", "official" in norm, f"got: {norm}")


# ===========================================================================
# locate_quote tests
# ===========================================================================
def test_locate_exact():
    print("\n═══ locate_quote: exact match ═══")
    chunk = "The Punjab Enforcement and Regulatory Authority was established in 2019."
    needle = "Punjab Enforcement and Regulatory Authority"
    result = locate_quote(chunk, needle)
    check("found", result is not None)
    if result:
        quote, start, end = result
        check("offset correct", chunk[start:end] == quote,
              f"chunk[{start}:{end}] = '{chunk[start:end]}' vs quote = '{quote}'")
        check("contains needle words", "punjab" in quote.lower() and "authority" in quote.lower())


def test_locate_soft_hyphen():
    print("\n═══ locate_quote: soft hyphen in chunk ═══")
    chunk = "The En\u00ADforce\u00ADment Officer has powers under Section 10."
    needle = "Enforcement Officer has powers"
    result = locate_quote(chunk, needle)
    check("found despite soft hyphens", result is not None)
    if result:
        quote, start, end = result
        check("offset correct", chunk[start:end] == quote)


def test_locate_line_break_hyphen():
    print("\n═══ locate_quote: line-break hyphen ═══")
    chunk = "The enforce-\nment of environ-\nmental regulations is a priority."
    needle = "enforcement of environmental regulations"
    result = locate_quote(chunk, needle)
    check("found despite hyphenation", result is not None)
    if result:
        quote, start, end = result
        check("offset correct", chunk[start:end] == quote)


def test_locate_extra_whitespace():
    print("\n═══ locate_quote: extra whitespace ═══")
    chunk = "Salary   and    Benefits:  SPPP-1    Rs. 350,082   per   month."
    needle = "Salary and Benefits: SPPP-1 Rs. 350,082"
    result = locate_quote(chunk, needle)
    check("found despite whitespace", result is not None)
    if result:
        quote, start, end = result
        check("offset correct", chunk[start:end] == quote)


def test_locate_table_text():
    print("\n═══ locate_quote: table-formatted text ═══")
    chunk = (
        "SPPP-1 | 350,082 | 1,000,082\n"
        "SPPP-2 | 275,000 | 800,000\n"
        "SPPP-3 | 200,000 | 600,000"
    )
    needle = "SPPP-1 | 350,082 | 1,000,082"
    result = locate_quote(chunk, needle)
    check("found in table", result is not None)
    if result:
        quote, start, end = result
        check("offset correct", chunk[start:end] == quote)


def test_locate_no_match():
    print("\n═══ locate_quote: no match ═══")
    chunk = "The PERA Act was enacted by the Punjab Assembly."
    needle = "JavaScript async await promises"
    result = locate_quote(chunk, needle)
    check("returns None for unrelated", result is None)


def test_locate_empty_inputs():
    print("\n═══ locate_quote: empty inputs ═══")
    check("empty chunk", locate_quote("", "hello") is None)
    check("empty needle", locate_quote("some text", "") is None)
    check("both empty", locate_quote("", "") is None)


def test_locate_nfkc_ligature():
    print("\n═══ locate_quote: NFKC ligature in chunk ═══")
    chunk = "The of\ufb01cial regulations state that compliance is mandatory."
    needle = "official regulations state"
    result = locate_quote(chunk, needle)
    check("found with ligature normalization", result is not None)
    if result:
        quote, start, end = result
        check("offset correct", chunk[start:end] == quote)


def test_locate_random_whitespace_fuzz():
    """Fuzz test: inject random whitespace into known text and verify localization."""
    print("\n═══ locate_quote: random whitespace fuzz ═══")
    base = "minimum monthly salary is Rs 350082 for SPPP grade one"
    words = base.split()

    successes = 0
    for trial in range(10):
        # Random whitespace injection
        parts = []
        for w in words:
            spaces = " " * random.randint(1, 5)
            if random.random() < 0.3:
                spaces = "\t" + spaces
            if random.random() < 0.2:
                spaces = "\n" + spaces
            parts.append(w + spaces)
        chunk = "".join(parts).strip()

        needle = "salary is Rs 350082 for SPPP"
        result = locate_quote(chunk, needle)
        if result:
            quote, start, end = result
            if chunk[start:end] == quote:
                successes += 1

    check(f"fuzz: {successes}/10 trials found with correct offsets", successes >= 7,
          f"got {successes}")


# ===========================================================================
# extract_table_row tests
# ===========================================================================
def test_table_row_sppp():
    print("\n═══ extract_table_row: SPPP row ═══")
    table = (
        "Grade | Minimum | Maximum\n"
        "SPPP-1 350,082 1,000,082\n"
        "SPPP-2 275,000 800,000\n"
        "SPPP-3 200,000 600,000"
    )
    rows = extract_table_row("sppp-1 salary", table)
    check("found SPPP-1 row", len(rows) >= 1)
    if rows:
        check("quote contains 350,082", "350,082" in rows[0]["quote"])
        check("has offsets", rows[0].get("start", -1) >= 0)


def test_table_row_generic():
    print("\n═══ extract_table_row: generic schedule ═══")
    table = (
        "Schedule-III SPPP Pay Scale\n"
        "SPPP-1 350,082 1,000,082\n"
        "SPPP-2 275,000 800,000\n"
    )
    rows = extract_table_row("schedule iii", table)
    check("returns header + 2 rows", len(rows) >= 2, f"got {len(rows)}")


def test_table_row_no_match():
    print("\n═══ extract_table_row: no match ═══")
    table = "Some random non-table text about PERA."
    rows = extract_table_row("sppp-5 salary", table)
    check("no matching rows", len(rows) == 0)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("Quote Localization — Unit Tests")
    print("=" * 60)

    # normalize_with_map tests
    test_normalize_basic()
    test_normalize_soft_hyphen()
    test_normalize_line_break_hyphen()
    test_normalize_crlf_hyphen()
    test_normalize_whitespace_collapse()
    test_normalize_nfkc()

    # locate_quote tests
    test_locate_exact()
    test_locate_soft_hyphen()
    test_locate_line_break_hyphen()
    test_locate_extra_whitespace()
    test_locate_table_text()
    test_locate_no_match()
    test_locate_empty_inputs()
    test_locate_nfkc_ligature()
    test_locate_random_whitespace_fuzz()

    # extract_table_row tests
    test_table_row_sppp()
    test_table_row_generic()
    test_table_row_no_match()

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("ALL PASS ✅")
    else:
        print("FAILURES ❌")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
