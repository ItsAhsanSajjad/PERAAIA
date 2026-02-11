"""Scan chunks for schedule/JD/Act content - temp diagnostic."""
from index_store import load_index_and_chunks
from retriever import _resolve_index_dir

_, chunks = load_index_and_chunks(_resolve_index_dir(None))
docs = {}
for c in chunks:
    dn = c.get("doc_name", "?")
    if dn not in docs:
        docs[dn] = 0
    docs[dn] += 1

with open("tmp_docs.txt", "w", encoding="utf-8") as f:
    for d in sorted(docs.keys()):
        f.write(f"[{docs[d]:3d}] {d}\n")
    f.write(f"\nTotal: {len(chunks)} chunks, {len(docs)} docs\n")

    # Schedule chunks
    f.write("\n=== SCHEDULE chunks ===\n")
    for i, c in enumerate(chunks):
        t = (c.get("text", "") or "").lower()
        dn = (c.get("doc_name", "") or "").lower()
        if "schedule" in t and ("sppp" in t or "pay scale" in t or "bps" in t):
            txt = c.get("text", "")[:300].replace("\n", "\\n")
            f.write(f"  chunk[{i}] doc={c.get('doc_name','?')} p={c.get('page_start',0)} len={len(c.get('text',''))}\n")
            f.write(f"    TEXT: {txt}\n\n")

    # JD chunks
    f.write("\n=== JD chunks ===\n")
    for i, c in enumerate(chunks):
        t = (c.get("text", "") or "").lower()
        if "position title" in t or "areas of responsibilities" in t:
            txt = c.get("text", "")[:300].replace("\n", "\\n")
            f.write(f"  chunk[{i}] doc={c.get('doc_name','?')} p={c.get('page_start',0)} len={len(c.get('text',''))}\n")
            f.write(f"    TEXT: {txt}\n\n")

    # Act section chunks
    f.write("\n=== ACT SECTION chunks ===\n")
    for i, c in enumerate(chunks):
        t = (c.get("text", "") or "").lower()
        dn = (c.get("doc_name", "") or "").lower()
        if ("act" in dn or "enforcement" in dn) and ("section" in t and not "job" in dn):
            txt = c.get("text", "")[:300].replace("\n", "\\n")
            f.write(f"  chunk[{i}] doc={c.get('doc_name','?')} p={c.get('page_start',0)} len={len(c.get('text',''))}\n")
            f.write(f"    TEXT: {txt}\n\n")

print("Written to tmp_docs.txt")
