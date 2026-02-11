"""Scan specific chunks for registry building."""
import sys
sys.path.insert(0, ".")
from index_store import load_index_and_chunks
from retriever import _resolve_index_dir

_, chunks = load_index_and_chunks(_resolve_index_dir(None))

with open("tmp_act.txt", "w", encoding="utf-8") as f:
    f.write("=== Bill/Act chunks ===\n")
    for i, c in enumerate(chunks):
        dn = c.get("doc_name", "") or ""
        if "Bill" in dn:
            txt = c.get("text", "")[:400].replace("\n", "\\n")
            f.write(f"chunk[{i}] p={c.get('page_start',0)} len={len(c.get('text',''))} TEXT={txt}\n\n")

    f.write("\n=== SSO chunks ===\n")
    for i in [367, 598]:
        if i < len(chunks):
            f.write(f"chunk[{i}] FULL TEXT:\n{chunks[i].get('text','')}\n\n===END===\n\n")

    f.write("\n=== Schedule-III chunk ===\n")
    for i in [404, 643]:
        if i < len(chunks):
            f.write(f"chunk[{i}] FULL TEXT:\n{chunks[i].get('text','')}\n\n===END===\n\n")

    f.write("\n=== CTO chunk ===\n")
    for i in [350, 581]:
        if i < len(chunks):
            f.write(f"chunk[{i}] FULL TEXT:\n{chunks[i].get('text','')[:1000]}\n\n===END===\n\n")

    f.write("\n=== Android chunk ===\n")
    for i in [359, 590]:
        if i < len(chunks):
            f.write(f"chunk[{i}] FULL TEXT:\n{chunks[i].get('text','')[:1000]}\n\n===END===\n\n")

    f.write("\n=== Manager Dev chunk ===\n")
    for i in [351, 582]:
        if i < len(chunks):
            f.write(f"chunk[{i}] FULL TEXT:\n{chunks[i].get('text','')[:1000]}\n\n===END===\n\n")

    f.write("\n=== Notification/Commencement/Establishment chunks ===\n")
    for i, c in enumerate(chunks):
        dn = (c.get("doc_name", "") or "").lower()
        if "commencement" in dn or "establishment" in dn:
            f.write(f"chunk[{i}] doc={c.get('doc_name','?')} TEXT:\n{c.get('text','')}\n\n===END===\n\n")

print("Written to tmp_act.txt")
