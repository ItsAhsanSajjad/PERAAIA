"""Find DG name chunk and directly check FAISS similarity."""
import numpy as np
from index_store import load_index_and_chunks

idx, chunks = load_index_and_chunks()

out = []
dg_positions = []
for i, c in enumerate(chunks):
    text = c.get("text", "")
    if "farrukh" in text.lower() or "atiq" in text.lower():
        dg_positions.append(i)
        out.append(f"Chunk {i}: {c.get('doc_name','')} page={c.get('loc_start','?')}")
        out.append(f"  text: {text[:300]}")
        out.append("")

out.append(f"Total DG name chunks: {len(dg_positions)}")

# Use openai to get embeddings
if dg_positions:
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    test_queries = ["director general name", "DG KON HAI", "who is dg pera", "name of dg"]
    for q in test_queries:
        resp = client.embeddings.create(input=[q], model=model)
        emb = resp.data[0].embedding
        emb_np = np.array([emb], dtype="float32")
        scores, indices = idx.search(emb_np, 60)
        
        for dg_i in dg_positions:
            if dg_i in indices[0]:
                rank = list(indices[0]).index(dg_i) + 1
                score = scores[0][list(indices[0]).index(dg_i)]
                out.append(f"Query '{q}': chunk {dg_i} at rank {rank}, score={score:.4f}")
            else:
                out.append(f"Query '{q}': chunk {dg_i} NOT in top 60!")

with open("debug_dg_faiss.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done")
