"""
Helper utilities for semantic retrieval refinements.

This module provides small, focused helpers used by the patched
agent to improve retrieval behavior:

- Intent detection for exclusive category filtering (Item/Pokemon/Move)
- Candidate filtering by category based on KB line headers
- Probable ID extraction from natural language queries
- KB path resolution and exact line lookup fallback

These functions keep `semantically_contextualize()` concise and make
new behaviors easier to test and maintain.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Set, Tuple
from difflib import SequenceMatcher

# Constants
CATEGORY_TOKENS = ("item", "pokemon", "pokémon", "move", "moves")
LEAD_IN_TOKENS = (":", "-", "about", "information", "info", "on", "of", "the", "named", "called")
KB_DEFAULT_PATH = os.path.join("data", "processed", "pokemon_kb.txt")
KB_HEADER_SEP = " — "


def intent_from_tokens(msg_tokens: List[str]) -> Tuple[bool, bool, bool, Set[str]]:
    """
    Determine intent flags and allowed categories from tokens.

    Args:
        msg_tokens: Lower-cased tokenized query.

    Returns:
        wants_item, wants_pokemon, wants_move, allowed_categories
    """
    wants_item = "item" in msg_tokens
    wants_pokemon = ("pokemon" in msg_tokens) or ("pokémon" in msg_tokens)
    wants_move = ("move" in msg_tokens) or ("moves" in msg_tokens)
    allowed: Set[str] = set()
    if wants_item:
        allowed.add("item")
    if wants_pokemon:
        allowed.add("pokemon")
    if wants_move:
        allowed.add("move")
    return wants_item, wants_pokemon, wants_move, allowed


def filter_candidates_by_category(results: Optional[List[dict]], allowed: Set[str]) -> Optional[List[dict]]:
    """
    Filter query results to only those whose KB header category is in `allowed`.

    KB lines follow the format: "<Category> — <Id> — <...>".

    Args:
        results: List of semantic DB result dicts with a `text` field.
        allowed: Set of allowed categories (lowercase), e.g., {"item"}.

    Returns:
        Filtered list if any matched; otherwise None to indicate no filtering applied.
    """
    if not results or not allowed:
        return None

    filtered: List[dict] = []
    for item in results:
        text = (item.get("text") or "").strip()
        parts = text.split(KB_HEADER_SEP)
        cat = (parts[0].strip().lower() if len(parts) >= 1 else "")
        if cat in allowed:
            filtered.append(item)

    return filtered if filtered else None


def candidate_labels(results: Optional[List[dict]]) -> List[str]:
    """
    Build compact debug labels in the form "<Category> — <Id>" for each result.

    Args:
        results: List of semantic DB result dicts with a `text` field.

    Returns:
        List of labels for debug printing.
    """
    labels: List[str] = []
    for item in results or []:
        text = (item.get("text") or "").strip()
        parts = text.split(KB_HEADER_SEP)
        if len(parts) >= 2:
            labels.append(f"{parts[0].strip()} — {parts[1].strip()}")
    return labels


def extract_probable_id(normalized_msg: str, msg_tokens: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a probable entity ID (e.g., "Pep-Up Plant") that follows a category
    token like "item", "pokemon", or "move" in the query.

    Heuristic: use substring after the first occurrence of any category token,
    then strip common lead-ins (":", "about", etc.).

    Args:
        normalized_msg: Entire query in lowercase.
        msg_tokens: Tokenized lowercase query.

    Returns:
        (probable_id, category_token_used)
    """
    cat_tok: Optional[str] = None
    for token in CATEGORY_TOKENS:
        if token in msg_tokens:
            cat_tok = token
            break
    if cat_tok is None:
        return None, None

    pos = normalized_msg.find(cat_tok)
    tail = normalized_msg[pos + len(cat_tok):].strip()
    for lead in LEAD_IN_TOKENS:
        tail = tail.lstrip().lstrip(lead).strip()

    probable_id = tail or None
    return probable_id, cat_tok


def kb_header_from_flags(wants_item: bool, wants_pokemon: bool, wants_move: bool) -> str:
    """
    Map intent flags to KB header strings.

    Returns:
        "item" | "pokemon" | "move"
    """
    if wants_item:
        return "item"
    if wants_pokemon:
        return "pokemon"
    return "move"


def select_best_by_id_similarity(results: Optional[List[dict]], normalized_msg: str, msg_tokens: List[str]) -> Tuple[Optional[dict], float]:
    """
    Choose the single best result by comparing the query to each entry's header `id`.

    Similarity rules:
    - Exact match: 1.0
    - Substring: 0.99
    - Token intersection: 0.95
    - Else: `SequenceMatcher` ratio against full message and per-token max

    Tie-breaker:
    - If scores are equal, prefer smaller `distance` (closer semantic match) when available.

    Fallback:
    - If no candidate improves the initial score and results are present, return the first.

    Args:
        results: List of semantic DB result dicts with `text` and optional `distance`.
        normalized_msg: Entire query in lowercase.
        msg_tokens: Tokenized lowercase query.

    Returns:
        (best_item_or_none, best_score)
    """
    best: Optional[dict] = None
    best_score: float = -1.0

    for item in results or []:
        text = (item.get("text") or "").strip()
        parts = text.split(KB_HEADER_SEP)
        candidate_id = (parts[1].strip().lower() if len(parts) >= 2 else "")

        # Compute similarity between the message and the candidate 'id' only
        score = 0.0
        if candidate_id:
            if candidate_id == normalized_msg:
                score = 1.0
            elif candidate_id in normalized_msg:
                score = 0.99
            elif any(tok in msg_tokens for tok in candidate_id.split()):
                score = 0.95
            else:
                score = SequenceMatcher(None, normalized_msg, candidate_id).ratio()
                for tok in msg_tokens:
                    score = max(score, SequenceMatcher(None, tok, candidate_id).ratio())

        if score > best_score:
            best_score = score
            best = item
        elif score == best_score and best is not None:
            try:
                d_cur = float(item.get("distance")) if item.get("distance") is not None else float("inf")
                d_best = float(best.get("distance")) if best.get("distance") is not None else float("inf")
                if d_cur < d_best:
                    best = item
            except Exception:
                pass

    if best is None and results:
        best = results[0]

    return best, best_score


def resolve_kb_path(semantic_where: Optional[dict]) -> str:
    """
    Resolve the KB file path from `semantic_where` (specifically `doc_name`),
    falling back to the default KB file.

    Args:
        semantic_where: Optional dict containing metadata filter with `doc_name`.

    Returns:
        Absolute or workspace-relative path to the KB file to read.
    """
    kb_path: Optional[str] = None
    if isinstance(semantic_where, dict):
        doc_name = semantic_where.get("doc_name")
        if isinstance(doc_name, str) and doc_name:
            kb_path = os.path.join("data", "processed", doc_name)
    return kb_path or KB_DEFAULT_PATH


def find_exact_kb_line(kb_path: str, kb_header: str, probable_id: Optional[str]) -> Optional[str]:
    """
    Find an exact KB line that starts with "<header> — <probable_id> —" (case-insensitive).

    Args:
        kb_path: Path to the KB text file.
        kb_header: Lowercase header ("item", "pokemon", "move").
        probable_id: Lowercase name to match after the header.

    Returns:
        The exact KB line (original casing) if found, else None.
    """
    if not probable_id:
        return None

    target_prefix = f"{kb_header} — {probable_id.strip().lower()} —"
    try:
        with open(kb_path, "r") as f:
            for line in f:
                if line.strip().lower().startswith(target_prefix):
                    return line.strip()
    except Exception:
        return None
    return None
