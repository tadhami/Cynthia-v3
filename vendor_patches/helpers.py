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
import re
import unicodedata
from unidecode import unidecode
from rapidfuzz import fuzz

# Constants
CATEGORY_TOKENS = ("item", "pokemon", "pokémon", "move", "moves")
LEAD_IN_TOKENS = (":", "-", "about", "information", "info", "on", "of", "the", "named", "called")
KB_DEFAULT_PATH = os.path.join("data", "processed", "pokemon_kb.txt")
KB_HEADER_SEP = " — "
PUNCT_CHARS = ".,:;!?()[]{}'\"’“”`"
def normalize_query_text(message: Optional[str]) -> Tuple[str, List[str]]:
    """
    Normalize a user query for consistent downstream tokenization and matching.

    - Lowercases the string
    - Replaces common unicode dashes with ASCII dash and spaces around them
    - Collapses whitespace
    - Splits into tokens (keeps original punctuation for later stripping by helpers)

    Args:
        message: Raw user query (may be None).

    Returns:
        (normalized_msg, msg_tokens)

    Examples:
        "Pokemon: Piplup" → ("pokemon: piplup", ["pokemon:", "piplup"])  → intent detects "pokemon"
        "Item — Pep-Up Plant" → ("item - pep-up plant", ["item", "-", "pep-up", "plant"]) → ID extraction OK
    """
    s = (message or "").strip()
    # Unicode canonical compatibility decomposition + recomposition
    s = unicodedata.normalize("NFKC", s)
    # Transliterate accents (Pokémon -> Pokemon), then case-fold for robust matching
    s = unidecode(s).casefold()
    # Normalize various dashes to a space-dash-space to aid tokenization
    s = s.replace("—", " - ").replace("–", " - ")
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    tokens = [t for t in s.split(" ") if t]
    return s, tokens



def intent_from_tokens(msg_tokens: List[str]) -> Tuple[bool, bool, bool, Set[str]]:
    """
    Determine intent flags and allowed categories from tokens.

    Args:
        msg_tokens: Lower-cased tokenized query.

    Returns:
        wants_item, wants_pokemon, wants_move, allowed_categories
    """
    # Normalize tokens by stripping surrounding punctuation so tokens like
    # "pokemon," or "item:" still map to the category keywords.
    tokens_norm = [t.strip(PUNCT_CHARS) for t in (msg_tokens or [])]
    token_set = set(tokens_norm)

    wants_item = "item" in token_set  # e.g., ["information","about","the","item","pep-up","plant"] → True
    wants_pokemon = ("pokemon" in token_set) or ("pokémon" in token_set)  # e.g., ["pokemon","piplup"] → True
    wants_move = ("move" in token_set) or ("moves" in token_set)  # e.g., ["move","ice","beam"] → True
    allowed: Set[str] = set()
    if wants_item:  # e.g., allowed becomes {"item"}
        allowed.add("item")
    if wants_pokemon:  # e.g., allowed becomes {"pokemon"}
        allowed.add("pokemon")
    if wants_move:  # e.g., allowed becomes {"move"}
        allowed.add("move")
    return wants_item, wants_pokemon, wants_move, allowed


# Removed: filter_candidates_by_category — category filtering is no longer applied post-query.


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
        text = (item.get("text") or "").strip()  # e.g., "Item — Super Repel — ..."
        parts = text.split(KB_HEADER_SEP)
        if len(parts) >= 2:
            labels.append(f"{parts[0].strip()} — {parts[1].strip()}")  # → "Item — Super Repel"
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
    for token in CATEGORY_TOKENS:  # e.g., finds "item" in "information about the item pep-up plant"
        # Accept raw or punct-stripped membership (handles "pokemon:" etc.)
        if token in msg_tokens or token in [t.strip(PUNCT_CHARS) for t in msg_tokens]:
            cat_tok = token
            break
    if cat_tok is None:
        return None, None

    # Locate the first occurrence of a category word using word boundaries.
    # Why word boundaries? They ensure we match whole words like "move" but not "remove".
    # The search is case-insensitive and robust to punctuation adjacent to the word (e.g., "pokemon:" or "Item —").
    # Examples that will match and set pos just after the category token:
    #   - "pokemon: piplup"      → boundary match on "pokemon" (pos points after "pokemon")
    #   - "Item — Pep-Up Plant"  → boundary match on "item" (pos points after "item")
    #   - "info on the move Ice Beam" → boundary match on "move" (pos after "move")
    m = re.search(r"\b(item|pokemon|pokémon|move|moves)\b", normalized_msg, flags=re.IGNORECASE)
    if m:
        # Start slicing immediately after the matched category token.
        pos = m.end()
    else:
        # Fallback: use the literal token position (covers rare cases where regex fails)
        pos = normalized_msg.find(cat_tok)

    # Grab everything after the category token. This should contain the human-given name/ID,
    # typically preceded by a small lead-in like ":" or "-" or words like "about", "named", etc.
    # Example:
    #   ": ice beam" → "ice beam"
    #   " - pep-up plant" → "pep-up plant"
    tail = normalized_msg[pos:].strip()

    # Strip any number of lead-in tokens at the beginning of the tail (as whole tokens).
    # The pattern means: from start (^) consume one-or-more (+) of the following, each optionally followed by spaces:
    #   - a punctuation token ":" or "-"
    #   - or one of the words: about, information, info, on, of, the, named, called
    # This is case-insensitive so variants like "Information:" or "ABOUT" are handled.
    # We apply it repeatedly until no further change so sequences like ": info on the" are fully removed.
    lead_pattern = r"^(?:(?::|\-|about|information|info|on|of|the|named|called)\s*)+"
    prev = None
    while prev != tail:
        prev = tail
        tail = re.sub(lead_pattern, "", tail, flags=re.IGNORECASE)

    # At this point, tail should be the clean, human-readable ID we want to match in the KB header.
    # Examples of final results:
    #   - Query: "information about the item pep-up plant" → tail: "pep-up plant"
    #   - Query: "Pokemon: Piplup" → tail: "piplup"
    #   - Query: "move - Ice Beam" → tail: "ice beam"
    probable_id = tail or None
    # Return the probable id and the category token detected so callers know which header to target.
    return probable_id, cat_tok  # e.g., ("pep-up plant","item") | ("piplup","pokemon") | ("ice beam","move")


def kb_header_from_flags(wants_item: bool, wants_pokemon: bool, wants_move: bool) -> str:
    """
    Map intent flags to KB header strings.

    Returns:
        "item" | "pokemon" | "move"
    """
    if wants_item:  # e.g., True,False,False → "item"
        return "item"
    if wants_pokemon:  # e.g., False,True,False → "pokemon"
        return "pokemon"
    return "move"  # e.g., False,False,True → "move"


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
        text = (item.get("text") or "").strip()  # e.g., "Item — Antidote — ..." | "Item — Pep-Up Plant — ..."
        parts = text.split(KB_HEADER_SEP)
        candidate_id = (parts[1].strip().lower() if len(parts) >= 2 else "")  # "antidote" | "pep-up plant"

        # Compute similarity between the message and the candidate 'id' only
        score = 0.0
        if candidate_id:
            # Use rapidfuzz for robust fuzzy scoring across token order and spacing
            # token_set_ratio handles multi-word comparisons well; scale to [0,1]
            rf_score = fuzz.token_set_ratio(normalized_msg, candidate_id) / 100.0
            # Capture strong substring relationships
            rf_partial = fuzz.partial_ratio(normalized_msg, candidate_id) / 100.0
            score = max(rf_score, rf_partial)

        if score > best_score:
            best_score = score
            best = item
        elif score == best_score and best is not None:
            try:
                d_cur = float(item.get("distance")) if item.get("distance") is not None else float("inf")  # smaller is better
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
            kb_path = os.path.join("data", "processed", doc_name)  # e.g., doc_name="pokemon_kb.txt"
    return kb_path or KB_DEFAULT_PATH  # default → data/processed/pokemon_kb.txt


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

    target_prefix = f"{kb_header} — {probable_id.strip().lower()} —"  # e.g., "item — pep-up plant —"
    try:
        with open(kb_path, "r") as f:  # e.g., data/processed/pokemon_kb.txt
            for line in f:
                if line.strip().lower().startswith(target_prefix):  # matches "Item — Pep-Up Plant — ..."
                    return line.strip()  # return the full exact KB line
    except Exception:
        return None
    return None
