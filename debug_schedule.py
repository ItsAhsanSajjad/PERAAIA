"""Check if Schedule (list of laws) exists in index chunks."""
from index_store import load_index_and_chunks
_, chunks = load_index_and_chunks()

# Search for schedule-related chunks
out = []
for i, c in enumerate(chunks):
    text = (c.get("text","") or "").lower()
    if "schedule" in text and ("law" in text or "act" in text):
        # Check if it looks like the Schedule table of laws
        if any(kw in text for kw in ["food", "motor", "canal", "drug", "local government", "environmental"]):
            out.append(f"SCHEDULE CHUNK {i}: {c.get('doc_name','')[:40]} p.{c.get('loc_start','?')}")
            out.append(f"  {c.get('text','')[:300]}")
            out.append("")

if not out:
    out.append("NO Schedule (list of specific laws) found in index!")
    # Try broader search
    out.append("\nBroadest match (any chunk with 'schedule' in first 200 chars):")
    cnt = 0
    for i, c in enumerate(chunks):
        text = (c.get("text","") or "")
        if "schedule" in text.lower()[:200] and cnt < 10:
            out.append(f"  Chunk {i}: {c.get('doc_name','')[:40]} p.{c.get('loc_start','?')}")
            out.append(f"    {text[:150]}")
            cnt += 1

with open("debug_schedule.txt","w",encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
