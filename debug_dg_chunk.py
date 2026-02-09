"""Show DG name chunks content."""
from index_store import load_index_and_chunks
idx, chunks = load_index_and_chunks()
for i, c in enumerate(chunks):
    text = c.get("text", "")
    if "farrukh" in text.lower():
        print(f"=== Chunk {i} | {c.get('doc_name','')} | page {c.get('loc_start','')} ===")
        print(text[:400])
        print()
