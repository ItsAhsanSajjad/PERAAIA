"""Write chunk report to file for reading."""
import json, os

active = json.load(open('assets/indexes/ACTIVE.json', 'r', encoding='utf-8'))
idx = active['active_index_dir']
chunks = []
with open(os.path.join(idx, 'chunks.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

out = []
out.append(f"Total chunks: {len(chunks)}")
out.append(f"Keys: {list(chunks[0].keys())}")
out.append(f"Sample chunk[0]:")
for k, v in chunks[0].items():
    val = str(v)[:100].encode('ascii', 'replace').decode()
    out.append(f"  {k}: {val}")

# Schedule-I
out.append("\n=== Schedule-I / Offences ===")
count = 0
for i, c in enumerate(chunks):
    t = (c.get('text') or c.get('embed_text_preview') or '').lower()
    if 'schedule-i' in t[:300] or ('offence' in t and 'schedule' in t[:200]):
        count += 1
        doc = c.get('doc_name', '?')
        raw = (c.get('text') or c.get('embed_text_preview') or '')[:300]
        preview = raw.encode('ascii', 'replace').decode()
        out.append(f"  [{i}] doc={doc[:60]}")
        out.append(f"    {preview}")
        out.append("")
        if count >= 3: break

# Schedule-III / SPPP
out.append("\n=== Schedule-III / SPPP ===")
count = 0
for i, c in enumerate(chunks):
    t = (c.get('text') or c.get('embed_text_preview') or '').lower()
    if 'sppp' in t and ('350' in t or 'schedule-iii' in t or 'minimum' in t):
        count += 1
        doc = c.get('doc_name', '?')
        raw = (c.get('text') or c.get('embed_text_preview') or '')[:300]
        preview = raw.encode('ascii', 'replace').decode()
        out.append(f"  [{i}] doc={doc[:60]}")
        out.append(f"    {preview}")
        out.append("")
        if count >= 4: break

# Job titles
out.append("\n=== Job Title Chunks ===")
titles = ['manager (development)', 'manager development', 'system support officer', 
          'chief technology officer', 'assistant manager gis', 'assistant manager (gis)',
          'cto']
for title in titles:
    matches = []
    for i, c in enumerate(chunks):
        t = (c.get('text') or c.get('embed_text_preview') or '').lower()
        if title in t:
            matches.append((i, c))
    out.append(f"  '{title}': {len(matches)} chunks")
    for i, c in matches[:2]:
        raw = (c.get('text') or c.get('embed_text_preview') or '')[:120]
        preview = raw.encode('ascii', 'replace').decode()
        out.append(f"    [{i}] doc={c.get('doc_name','?')[:40]} | {preview}")

# Manager Dev salary/SPPP
out.append("\n=== Manager (Dev) + salary/SPPP ===")
for i, c in enumerate(chunks):
    t = (c.get('text') or c.get('embed_text_preview') or '').lower()
    if ('manager' in t and 'development' in t) and ('salary' in t or 'sppp' in t or 'bps' in t):
        raw = (c.get('text') or c.get('embed_text_preview') or '')[:250]
        preview = raw.encode('ascii', 'replace').decode()
        out.append(f"  [{i}] doc={c.get('doc_name','?')[:40]} | {preview}")

with open('scripts/chunks_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))

print("Report written to scripts/chunks_report.txt")
