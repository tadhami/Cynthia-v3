#!/usr/bin/env python3
"""
Batch semantic query tests.

Initializes the PatchedAgent, loads the KB, then runs ~50 semantic queries with
variations in casing/punctuation and lead-in phrases. Writes a CSV with columns:
- semantic_query
- candidates (compact labels like "Item — Antidote")
- identified_candidate (chosen header id)

Output: reports/semantic_query_tests.csv
"""
import sys
import os
import csv

# Ensure workspace root is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vendor_patches.patch_noahs_agent import PatchedAgent
from vendor_patches.helpers import (
    normalize_query_text,
    intent_from_tokens,
    filter_candidates_by_category,
    candidate_labels,
    select_best_by_id_similarity,
    extract_probable_id,
    kb_header_from_flags,
    resolve_kb_path,
    find_exact_kb_line,
)


def build_queries() -> list[str]:
    q = []
    # Items (mix case and punctuation)
    items = [
        "pep-up plant", "antidote", "super repel", "charcoal", "white herb"
    ]
    for name in items:
        q += [
            f"information about the item {name}",
            f"Information about the Item {name}",
            f"item {name}",
            f"Item: {name}",
            f"Item — {name}",
            f"info on item {name}",
        ]
    # Pokémon
    pokes = [
        "piplup", "breloom", "seedot", "anorith", "forretress", "lileep"
    ]
    for name in pokes:
        q += [
            f"information about the pokemon {name}",
            f"Information about the Pokemon {name}",
            f"pokemon {name}",
            f"Pokemon: {name}",
            f"Pokémon {name}",
            f"info on pokemon {name}",
        ]
    # Moves (common ones; may miss if KB lacks entries — that's fine for review)
    moves = [
        "ice beam", "thunderbolt", "flamethrower", "earthquake", "psychic"
    ]
    for name in moves:
        q += [
            f"information about the move {name}",
            f"Information about the Move {name}",
            f"move {name}",
            f"Move: {name}",
            f"Move — {name}",
            f"info on move {name}",
        ]
    # Trim to ~50 diverse cases
    return q[:50]


def run_query(agent: PatchedAgent, query: str, top_k: int = 10, doc_name: str = "pokemon_kb.txt") -> tuple[str, list[str], str]:
    """
    Execute retrieval + selection for a single query, returning:
    (semantic_query, candidates_labels, identified_candidate_id)
    """
    normalized_msg, msg_tokens = normalize_query_text(query)
    where = {"doc_name": doc_name}
    results = agent.semantic_db.query(query, top_k=top_k, where=where)

    wants_item, wants_pokemon, wants_move, allowed = intent_from_tokens(msg_tokens)
    maybe_filtered = filter_candidates_by_category(results, allowed)
    used_results = maybe_filtered if maybe_filtered is not None else results

    labels = candidate_labels(used_results)

    best, best_score = select_best_by_id_similarity(used_results, normalized_msg, msg_tokens)
    chosen_id = None
    if best is not None:
        parts = (best.get("text") or "").strip().split(" — ")
        if len(parts) >= 2:
            chosen_id = parts[1].strip()

    # Exact-ID fallback
    try:
        wants_any = wants_item or wants_pokemon or wants_move
        weak_match = (best_score < 0.98) or (chosen_id is None)
        if wants_any and weak_match:
            probable_id, _ = extract_probable_id(normalized_msg, msg_tokens)
            kb_header = kb_header_from_flags(wants_item, wants_pokemon, wants_move)
            kb_path = resolve_kb_path(where)
            exact_line = find_exact_kb_line(kb_path, kb_header, probable_id)
            if exact_line:
                parts = exact_line.strip().split(" — ")
                if len(parts) >= 2:
                    chosen_id = parts[1].strip()
    except Exception:
        pass

    return query, labels, chosen_id or ""


def main():
    # Init agent and KB
    agent = PatchedAgent(name="Cynthia", model="llama3.2:3b")
    agent.semantic_db.purge_collection()
    agent.upload_document(os.path.join("data", "processed", "pokemon_kb.txt"), max_sentences_per_chunk=1, split_mode="lines")

    queries = build_queries()
    rows = []
    for q in queries:
        sq, labels, ident = run_query(agent, q)
        rows.append({
            "semantic_query": sq,
            "candidates": "; ".join(labels),
            "identified_candidate": ident,
        })

    out_path = os.path.join(ROOT, "reports", "semantic_query_tests.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["semantic_query", "candidates", "identified_candidate"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
