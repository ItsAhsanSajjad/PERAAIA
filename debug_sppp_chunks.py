"""Debug: Find all SPPP-related chunks to understand data structure."""
from index_store import load_index_and_chunks
import re, json

idx, chunks = load_index_and_chunks()

results = []
for i, c in enumerate(chunks):
    text = c.get("text", "")
    if "SPPP" in text and re.search(r'\d{2,3},\d{3}', text):
        results.append({
            "idx": i,
            "doc": c.get("doc_name", "?"),
            "page": c.get("loc_start", "?"),
            "text_preview": text[:400]
        })

with open("debug_sppp_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Found {len(results)} chunks. Written to debug_sppp_output.json")
