#!/usr/bin/env python3
"""
Quick smoke tests for vendor_patches.helpers.

Runs lightweight checks for:
- normalize_query_text
- intent_from_tokens
- extract_probable_id
- resolve_kb_path
- find_exact_kb_line (requires data/processed/pokemon_kb.txt)

Usage:
  python scripts/helpers_smoke_test.py
"""
import sys
import os

# Ensure workspace root is on sys.path so `vendor_patches` is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vendor_patches import helpers as H


def assert_true(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def test_normalize_and_intent():
    s, toks = H.normalize_query_text("Pokemon: Piplup")
    wants_item, wants_pokemon, wants_move, _ = H.intent_from_tokens(toks)
    assert_true(wants_pokemon and not wants_item and not wants_move, "intent_from_tokens pokemon failed")

    s, toks = H.normalize_query_text("item — Pep-Up Plant")
    wants_item, wants_pokemon, wants_move, _ = H.intent_from_tokens(toks)
    assert_true(wants_item, "intent_from_tokens item failed")

    s, toks = H.normalize_query_text("MOVE Ice Beam")
    wants_item, wants_pokemon, wants_move, _ = H.intent_from_tokens(toks)
    assert_true(wants_move, "intent_from_tokens move failed")


def test_probable_id():
    s, toks = H.normalize_query_text("information about the item Pep-Up Plant")
    pid, cat = H.extract_probable_id(s, toks)
    assert_true(pid == "pep-up plant" and cat in ("item", "pokemon", "pokémon", "move", "moves"), f"extract_probable_id item failed: {pid}, {cat}")

    s, toks = H.normalize_query_text("Pokemon Piplup")
    pid, cat = H.extract_probable_id(s, toks)
    assert_true(pid == "piplup" and "pokemon" in cat, f"extract_probable_id pokemon failed: {pid}, {cat}")

    s, toks = H.normalize_query_text("move: Ice Beam")
    pid, cat = H.extract_probable_id(s, toks)
    assert_true(pid == "ice beam" and "move" in cat, f"extract_probable_id move failed: {pid}, {cat}")


def test_exact_line_lookup():
    s, toks = H.normalize_query_text("Item Pep-Up Plant")
    pid, _ = H.extract_probable_id(s, toks)
    kb_path = H.resolve_kb_path(None)
    line = H.find_exact_kb_line(kb_path, "item", pid)
    assert_true(line is not None and line.lower().startswith("item — pep-up plant —"), "find_exact_kb_line Pep-Up Plant failed")


def main():
    test_normalize_and_intent()
    test_probable_id()
    test_exact_line_lookup()
    print("All helper smoke tests passed.")


if __name__ == "__main__":
    main()
