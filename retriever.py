from __future__ import annotations

import os
import re
from typing import List, Dict, Any
from collections import defaultdict

from index_store import load_index_and_chunks, embed_texts, _normalize_vectors

# -----------------------------
# Retrieval configuration
# -----------------------------
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "40"))
SIM_THRESHOLD = float(os.getenv("RETRIEVER_SIM_THRESHOLD", "0.20"))
MAX_CHUNKS_PER_DOC = int(os.getenv("RETRIEVER_MAX_CHUNKS_PER_DOC", "6"))

STRONG_SIM_THRESHOLD = float(os.getenv("RETRIEVER_STRONG_SIM_THRESHOLD", "0.28"))

MIN_KEYWORD_MATCHES = int(os.getenv("RETRIEVER_MIN_KEYWORD_MATCHES", "2"))
RELATIVE_DOC_SCORE_KEEP = float(os.getenv("RETRIEVER_RELATIVE_DOC_SCORE_KEEP", "0.80"))
MAX_DOCS_RETURNED = int(os.getenv("RETRIEVER_MAX_DOCS_RETURNED", "4"))

QUERY_VARIANTS_ENABLED = os.getenv("RETRIEVER_QUERY_VARIANTS_ENABLED", "1").strip() != "0"
MAX_QUERY_VARIANTS = int(os.getenv("RETRIEVER_MAX_QUERY_VARIANTS", "3"))

RELAXED_MIN_KEYWORD_MATCHES = int(os.getenv("RETRIEVER_RELAXED_MIN_KEYWORD_MATCHES", "1"))

LEX_FALLBACK_ENABLED = os.getenv("RETRIEVER_LEX_FALLBACK_ENABLED", "1").strip() != "0"
LEX_FALLBACK_MAX = int(os.getenv("RETRIEVER_LEX_FALLBACK_MAX", "80"))
LEX_FALLBACK_PER_DOC = int(os.getenv("RETRIEVER_LEX_FALLBACK_PER_DOC", "3"))

# Criteria precision
CRITERIA_DOC_PRIORITIZATION = os.getenv("RETRIEVER_CRITERIA_DOC_PRIORITIZATION", "1").strip() != "0"
CRITERIA_MIN_DOCS = int(os.getenv("RETRIEVER_CRITERIA_MIN_DOCS", "2"))  # keep at least 2 docs if possible


# -----------------------------
# Keyword extraction
# -----------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "from", "by", "with", "about", "tell", "me",
    "who", "what", "when", "where", "why", "how", "please",
    "pera",
}

# intent-ish / filler words that should NOT dominate overlap decisions
_INTENT_STOP = {
    "role", "roles", "duty", "duties", "function", "functions", "responsibility",
    "responsibilities", "tor", "tors", "term", "terms", "reference",
    "criteria", "criterion", "eligibility", "eligible", "qualification", "qualifications",
    "experience", "education", "minimum", "required", "requirement", "requirements",
    "position", "positions", "post", "posts", "job", "jobs", "main", "most", "power", "authority",
}

_KEEP_SHORT = {"ai", "ml", "it", "hr", "ppra", "ipo", "cto", "tor", "tors", "dg"}

# deterministic abbreviation expansions (retrieval-side)
_ABBREV_MAP = {
    "cto": "chief technology officer",
    "tor": "terms of reference",
    "tors": "terms of reference",
    "dg": "director general",
    "hr": "human resource",
    "it": "information technology",

    # common shorthand / informal
    "mgr": "manager",
    "dev": "development",
    "sr": "senior",
    "jr": "junior",
}


def _expand_abbrev(s: str) -> str:
    t = (s or "").lower()
    for k, v in _ABBREV_MAP.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t)
    return t


def _normalize_text(s: str) -> str:
    s = _expand_abbrev(s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _stem_token(t: str) -> str:
    """
    Very small deterministic stemmer (plural/singular robustness):
    - sergeants -> sergeant
    - positions -> position
    - policies -> policy
    """
    t = (t or "").strip().lower()
    if len(t) <= 3:
        return t

    if t.endswith("'s"):
        t = t[:-2]

    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 4:
        return t[:-1]

    return t


def _tokenize_for_overlap(s: str) -> List[str]:
    q = _normalize_text(s)
    toks: List[str] = []
    for raw in q.split():
        if not raw:
            continue
        if raw in _STOPWORDS:
            continue
        if len(raw) >= 3 or raw in _KEEP_SHORT:
            toks.append(_stem_token(raw))
    return toks


def _extract_keywords(question: str) -> List[str]:
    toks = _tokenize_for_overlap(question)
    seen = set()
    out: List[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:12]


def _entity_keywords(question: str) -> List[str]:
    """
    Entity-focused tokens (titles/roles/nouns) to avoid overlap being dominated
    by generic intent words like "criteria", "role", "authority", etc.
    """
    toks = _tokenize_for_overlap(question)
    ent: List[str] = []
    seen = set()
    for t in toks:
        if t in _INTENT_STOP:
            continue
        if t in seen:
            continue
        seen.add(t)
        ent.append(t)
    return ent[:10]


def _keyword_overlap_count(keywords: List[str], text: str) -> int:
    if not keywords:
        return 0
    kw_set = set(keywords)
    text_tokens = set(_tokenize_for_overlap(text))
    return len(kw_set.intersection(text_tokens))


def _rows_by_id(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        try:
            out[int(r.get("id", -1))] = r
        except Exception:
            continue
    return out


def _load_index_rows(index_dir: str):
    try:
        return load_index_and_chunks(index_dir=index_dir)  # type: ignore[arg-type]
    except TypeError:
        return load_index_and_chunks()


# -----------------------------
# Intent helpers
# -----------------------------
_COMPOSITION_PAT = re.compile(r"\b(composition|constitut|constitution|constitute|consist|comprise|members?|authority)\b", re.I)
_COMPOSITION_PHRASES = [
    "authority shall consist of",
    "constitution of the authority",
    "members of the authority",
    "the authority shall consist",
    "shall consist of the following members",
]

_CRITERIA_PAT = re.compile(
    r"\b(criteria|criterion|eligib|qualification|qualify|required|requirement|experience|education|minimum|degree|age|skills?)\b",
    re.I
)
_CRITERIA_PHRASES = [
    "eligibility criteria",
    "minimum qualification",
    "qualification and experience",
    "required qualification",
    "experience required",
    "education",
    "age limit",
    "skills",
]

# role / TOR intent
_ROLE_PAT = re.compile(r"\b(role|roles|tor|tors|terms of reference|duty|duties|responsibil|function|job description)\b", re.I)

# informal “most power / main authority” intent
_POWER_PAT = re.compile(r"\b(main authority|most power|most powerful|who holds the most power|who is powerful)\b", re.I)


def _intent_extra_keywords(question: str) -> List[str]:
    q = (question or "").lower()
    extras: List[str] = []

    if _COMPOSITION_PAT.search(q):
        extras.extend([
            "shall", "consist", "comprise", "constitution", "member",
            "chairperson", "vice", "secretary",
            "include", "following",
        ])

    if _CRITERIA_PAT.search(q):
        extras.extend([
            "eligibility", "criteria", "qualification", "experience",
            "education", "minimum", "required", "requirement",
            "age", "degree", "competenc", "skill",
        ])

    if _ROLE_PAT.search(q):
        extras.extend([
            "terms", "reference", "tor", "duties", "responsibilities",
            "functions", "report", "reports", "wing", "purpose",
        ])

    if _POWER_PAT.search(q):
        # common formal anchors (PERA docs often use formal titles)
        extras.extend([
            "chairperson", "vice", "authority", "director", "general", "member", "secretary",
        ])

    extras2: List[str] = []
    seen = set()
    for e in extras:
        se = _stem_token(e)
        if se in seen:
            continue
        seen.add(se)
        extras2.append(se)
    return extras2


def _swap_two_word_title(ent: List[str]) -> str:
    """
    If entity looks like two words (e.g., development manager),
    produce swapped order: manager development.
    """
    if len(ent) != 2:
        return ""
    a, b = ent[0], ent[1]
    if not a or not b:
        return ""
    return f"{b} {a}".strip()


def _build_query_variants(question: str) -> List[str]:
    """
    Systematic, deterministic query expansion to prevent regressions from:
    - word order changes ("Development Manager" vs "Manager Development")
    - informal phrasing ("main authority / most power")
    - role phrasing variations ("role" vs "duties" vs "TORs")
    """
    q = (question or "").strip()
    if not q:
        return [q]

    variants: List[str] = [q]
    qn = _normalize_text(q)

    # base entity tokens for title-like expansion
    ent = _entity_keywords(q)
    ent_phrase = " ".join(ent[:4]).strip()

    # composition / constitution expansions
    if _COMPOSITION_PAT.search(q):
        variants.append("Authority shall consist of the following members chairperson vice chairperson secretary member")
        variants.append("Constitution of the Authority members of the Authority")

    # criteria expansions
    if _CRITERIA_PAT.search(qn):
        variants.append(f"{q} eligibility criteria qualification experience")
        variants.append(f"{q} minimum qualification experience required")

    # role/TOR expansions
    if _ROLE_PAT.search(qn) and ent_phrase:
        variants.append(f"terms of reference of {ent_phrase} in PERA")
        variants.append(f"duties and responsibilities of {ent_phrase} in PERA")
        variants.append(f"job description of {ent_phrase} in PERA")

    # word-order robustness for 2-word role titles
    swapped = _swap_two_word_title(ent[:2])
    if swapped:
        # preserve original user query but add swap variants
        variants.append(re.sub(r"\s+", " ", q, flags=re.I).strip().replace(ent[0] + " " + ent[1], swapped))
        variants.append(f"{swapped} role duties responsibilities in PERA")

    # informal power/authority mapping
    if _POWER_PAT.search(qn):
        variants.append("chairperson vice chairperson authority powers")
        variants.append("director general powers functions authority")
        variants.append("who is chairperson of the authority and who has powers")

    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for v in variants:
        v = v.strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)

    return out[:MAX_QUERY_VARIANTS]


def _required_overlap(keywords: List[str], strict: bool) -> int:
    n = len(keywords)
    if n == 0:
        return 0
    if strict:
        if n == 1:
            return 1
        return max(2, MIN_KEYWORD_MATCHES)
    return max(1, RELAXED_MIN_KEYWORD_MATCHES)


def _lexical_fallback_hits(rows: List[Dict[str, Any]], all_keywords: List[str], entity_kw: List[str], question: str) -> Dict[int, float]:
    """
    Lexical fallback is ONLY a safety net when:
      - query is composition/criteria-like
      - AND the row text matches strong legal phrases or has meaningful overlap
      - AND at least one ENTITY keyword appears (prevents broad-intent noise)
    """
    if not LEX_FALLBACK_ENABLED:
        return {}

    qn = _normalize_text(question)
    want_comp = _COMPOSITION_PAT.search(qn) is not None
    want_criteria = _CRITERIA_PAT.search(qn) is not None
    if not want_comp and not want_criteria:
        return {}

    phrases = _COMPOSITION_PHRASES if want_comp else _CRITERIA_PHRASES

    best: Dict[int, float] = {}
    per_doc_counts: Dict[str, int] = defaultdict(int)

    for r in rows:
        if not r.get("active", True):
            continue

        text = r.get("text") or ""
        if not text:
            continue

        tn = _normalize_text(text)

        phrase_hit = any(p in tn for p in phrases)

        # overlap against all keywords (intent+entity)
        overlap_all = _keyword_overlap_count(all_keywords, text)

        # overlap against entity keywords (role/title tokens)
        overlap_ent = _keyword_overlap_count(entity_kw, text) if entity_kw else 0

        # ✅ Key regression fix: do not accept generic matches without entity anchor
        # (prevents “criteria” queries from pulling unrelated HR/panel docs)
        if entity_kw and overlap_ent < 1 and not phrase_hit:
            continue

        if not phrase_hit and overlap_all < 2:
            continue

        doc_name = r.get("doc_name", "Unknown document")
        if per_doc_counts[doc_name] >= LEX_FALLBACK_PER_DOC:
            continue

        try:
            cid = int(r.get("id"))
        except Exception:
            continue

        pseudo = 0.45 + min(0.20, 0.03 * overlap_all)
        if phrase_hit:
            pseudo = max(pseudo, 0.62)

        prev = best.get(cid)
        if prev is None or pseudo > prev:
            best[cid] = pseudo
            per_doc_counts[doc_name] += 1

        if len(best) >= LEX_FALLBACK_MAX:
            break

    return best


# Criteria doc scoring (precision)
def _criteria_doc_signal_score(doc: Dict[str, Any]) -> float:
    hits = doc.get("hits", []) or []
    if not hits:
        return 0.0

    h0 = hits[0]
    txt = _normalize_text(h0.get("text") or "")
    phrase_hits = 0
    for p in _CRITERIA_PHRASES:
        if p in txt:
            phrase_hits += 1

    score = float(h0.get("score", 0.0) or 0.0)
    overlap = int(h0.get("overlap", 0) or 0)

    return (phrase_hits * 10.0) + (overlap * 1.5) + (score * 1.0)


def _apply_criteria_doc_prioritization(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not CRITERIA_DOC_PRIORITIZATION:
        return evidence_docs

    qn = _normalize_text(question)
    if _CRITERIA_PAT.search(qn) is None:
        return evidence_docs

    scored = [(ed, _criteria_doc_signal_score(ed)) for ed in evidence_docs]
    scored.sort(key=lambda x: x[1], reverse=True)

    kept = [x[0] for x in scored[:max(CRITERIA_MIN_DOCS, 1)]]
    kept_names = {d.get("doc_name") for d in kept}

    for ed in evidence_docs:
        if ed.get("doc_name") in kept_names:
            continue
        kept.append(ed)
        kept_names.add(ed.get("doc_name"))
        if len(kept) >= MAX_DOCS_RETURNED:
            break

    return kept[:MAX_DOCS_RETURNED]


# -----------------------------
# Main retrieval
# -----------------------------
def retrieve(question: str, index_dir: str = "assets/index") -> Dict[str, Any]:
    empty = {
        "question": question,
        "has_evidence": False,
        "primary_doc": None,
        "primary_doc_rank": 0,
        "evidence": []
    }

    question = (question or "").strip()
    if not question:
        return empty

    try:
        idx, rows = _load_index_rows(index_dir=index_dir)
        if idx is None or not rows:
            return empty

        id_to_row = _rows_by_id(rows)

        queries = [question]
        if QUERY_VARIANTS_ENABLED:
            queries = _build_query_variants(question)

        # keywords:
        # - entity_kw: title/role tokens (robust to word order and pluralization)
        # - all_kw: entity + intent extras
        entity_kw = _entity_keywords(question)
        base_kw = _extract_keywords(question)
        extras_kw = _intent_extra_keywords(question)
        all_kw = base_kw + [k for k in extras_kw if k not in base_kw]

        # semantic search across variants
        q_vecs = embed_texts(queries)
        q_vecs = _normalize_vectors(q_vecs)
        scores_mat, ids_mat = idx.search(q_vecs, TOP_K)

        best_by_id: Dict[int, float] = {}

        for qi in range(len(queries)):
            scores = scores_mat[qi].tolist()
            ids = ids_mat[qi].tolist()
            for score, vid in zip(scores, ids):
                if vid == -1:
                    continue
                score = float(score)
                if score < SIM_THRESHOLD:
                    continue
                vid_i = int(vid)
                prev = best_by_id.get(vid_i)
                if prev is None or score > prev:
                    best_by_id[vid_i] = score

        # lexical fallback merge (composition/criteria only, entity-anchored)
        lex_best = _lexical_fallback_hits(rows, all_kw, entity_kw, question)
        for cid, pseudo in lex_best.items():
            prev = best_by_id.get(cid)
            if prev is None or pseudo > prev:
                best_by_id[cid] = pseudo

        if not best_by_id:
            return empty

        # Build hit objects
        hits: List[Dict[str, Any]] = []
        for vid_i, score in best_by_id.items():
            r = id_to_row.get(int(vid_i))
            if not r or not r.get("active", True):
                continue

            text = (r.get("text") or "")

            # ✅ Overlap should primarily reflect ENTITY alignment (prevents noisy intent-only matches)
            overlap_entity = _keyword_overlap_count(entity_kw, text) if entity_kw else 0
            overlap_all = _keyword_overlap_count(all_kw, text)

            # store overlap as entity overlap if we have entity tokens; else fall back to all overlap
            overlap = overlap_entity if entity_kw else overlap_all

            hits.append({
                "id": int(vid_i),
                "score": float(score),
                "overlap": int(overlap),
                "doc_name": r.get("doc_name", "Unknown document"),
                "doc_rank": int(r.get("doc_rank", 0) or 0),
                "text": text,
                "source_type": r.get("source_type", ""),
                "loc_kind": r.get("loc_kind", ""),
                "loc_start": r.get("loc_start"),
                "loc_end": r.get("loc_end"),
                "path": r.get("path"),
                # keep diagnostics (harmless if ignored)
                "overlap_all": int(overlap_all),
                "overlap_entity": int(overlap_entity),
            })

        if not hits:
            return empty

        # strict pass (use entity-based overlap gates)
        strict_req = _required_overlap(entity_kw if entity_kw else base_kw, strict=True)
        strict_hits = [h for h in hits if (int(h.get("overlap", 0)) >= strict_req)]

        if not strict_hits:
            relaxed_req = _required_overlap(entity_kw if entity_kw else base_kw, strict=False)
            strict_hits = [h for h in hits if (int(h.get("overlap", 0)) >= relaxed_req)]

        final_hits = strict_hits if strict_hits else hits

        # group by document
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        ranks: Dict[str, int] = {}
        for h in final_hits:
            dn = h["doc_name"]
            grouped[dn].append(h)
            ranks[dn] = max(ranks.get(dn, 0), int(h.get("doc_rank", 0) or 0))

        evidence_docs: List[Dict[str, Any]] = []
        for dn, doc_hits in grouped.items():
            doc_hits.sort(
                key=lambda x: (float(x.get("score", 0.0) or 0.0), int(x.get("overlap", 0) or 0)),
                reverse=True
            )
            doc_hits = doc_hits[:MAX_CHUNKS_PER_DOC]
            evidence_docs.append({
                "doc_name": dn,
                "doc_rank": ranks.get(dn, 0),
                "hits": doc_hits
            })

        def best_score(ed: Dict[str, Any]) -> float:
            hs = ed.get("hits", [])
            return float(hs[0]["score"]) if hs else 0.0

        def best_overlap(ed: Dict[str, Any]) -> int:
            hs = ed.get("hits", [])
            return int(hs[0].get("overlap", 0)) if hs else 0

        evidence_docs.sort(
            key=lambda ed: (ed.get("doc_rank", 0), best_score(ed), best_overlap(ed)),
            reverse=True
        )

        strong = [
            ed for ed in evidence_docs
            if ed.get("hits") and float(ed["hits"][0]["score"]) >= STRONG_SIM_THRESHOLD
        ]
        primary = strong[0] if strong else evidence_docs[0]

        best = best_score(primary)
        kept_docs: List[Dict[str, Any]] = []
        for ed in evidence_docs:
            if best <= 0:
                continue
            if best_score(ed) >= best * RELATIVE_DOC_SCORE_KEEP:
                kept_docs.append(ed)

        kept_docs = kept_docs[:MAX_DOCS_RETURNED]

        # apply criteria precision reorder (after pruning)
        kept_docs = _apply_criteria_doc_prioritization(question, kept_docs)

        return {
            "question": question,
            "has_evidence": True,
            "primary_doc": primary.get("doc_name"),
            "primary_doc_rank": int(primary.get("doc_rank", 0) or 0),
            "evidence": kept_docs
        }

    except Exception:
        return empty
