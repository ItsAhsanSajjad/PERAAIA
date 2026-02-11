"""Find SSO / System Support Officer in index chunks - write to file."""
from index_store import load_index_and_chunks
_, chunks = load_index_and_chunks()

out = []
terms = ["system support", "support officer", " sso "]
for i, c in enumerate(chunks):
    text = (c.get("text","") or "")
    text_l = text.lower()
    for t in terms:
        if t in f" {text_l} ":
            out.append(f"MATCH '{t}' -> Chunk {i}: {c.get('doc_name','')[:40]} p.{c.get('loc_start','?')}")
            out.append(f"  {text[:300]}")
            out.append("")
            break

out.insert(0, f"Total SSO matches: {len([l for l in out if l.startswith('MATCH')])}")
with open("find_sso_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
