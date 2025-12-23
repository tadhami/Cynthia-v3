from noahs_local_ollama_chat_agent.agent import ollama_chat_agent as BaseAgent
from noahs_local_ollama_chat_agent.local_sql_db import local_sql_db
from noahs_local_ollama_chat_agent.local_semantic_db import local_semantic_db
import os
import json
from difflib import SequenceMatcher
from vendor_patches.helpers import (
    intent_from_tokens,
    extract_probable_id,
    kb_header_from_flags,
    resolve_kb_path,
    find_exact_kb_line,
    candidate_labels,
    select_best_by_id_similarity,
    normalize_query_text,
)


class PatchedAgent(BaseAgent):
    def __init__(self, model="llama3.2:3b", model_encoding=None, name="Agent", url="http://localhost:11434/api", context_window_limit=2048, semantic_model_name="multi-qa-MiniLM-L6-cos-v1"):
        # Store semantic model choice used when initializing semantic DB (base agent lacks this param)
        self.semantic_model_name = semantic_model_name
        super().__init__(model=model, model_encoding=model_encoding, name=name, url=url, context_window_limit=context_window_limit)
    def _initialize_databases(self):
        """Initialize DBs; pass a specific sentence transformer name to semantic DB."""
        self.semantic_db = local_semantic_db(
            persist_directory=f"{self.name}_data/{self.name}_semantic_db",
            collection_name=f"{self.name}_general",
            # Difference vs base: explicitly control the sentence transformer via instance field
            sentence_transformer_name=getattr(self, "semantic_model_name", None) or "multi-qa-MiniLM-L6-cos-v1",
        )
        self.sql_db = local_sql_db(f"{self.name}_data/{self.name}_sql_db")

    def upload_document(self, doc_path, doc_name=None, max_sentences_per_chunk=5, metadata=None, split_mode="sentences"):
        if doc_name is None:
            # Same as default: derive a readable name for metadata from file path
            doc_name = os.path.basename(doc_path)
        with open(doc_path, "r") as file:
            doc = file.read()
        if metadata is None:
            # Same default key used downstream for filtering: {"doc_name": doc_name}
            metadata = {"doc_name": doc_name}
        if split_mode == "lines":
            # Difference vs default: manual line-based chunking instead of semantic splitter
            lines = doc.splitlines()
            texts, metadatas, text_ids = [], [], []
            idx = 0
            for i in range(0, len(lines), max_sentences_per_chunk):
                # Reuse max_sentences_per_chunk as "lines per chunk"
                chunk = "\n".join(lines[i:i + max_sentences_per_chunk])
                texts.append(chunk)
                # Clone base metadata per chunk and add an explicit chunk index (not in default)
                m = eval(repr(metadata))
                m["index"] = idx
                metadatas.append(m)
                # Difference vs default: assign stable per-chunk IDs ("<doc_name>-<index>")
                text_ids.append(f"{doc_name}-{idx}")
                idx += 1
            # Difference vs default: use batch_insert directly for pre-chunked texts
            # batch_insert when you’ve pre-chunked and need to preserve custom boundaries, metadata, and IDs; use insert_in_chunks when you want the semantic DB to handle splitting for you.
            self.semantic_db.batch_insert(texts=texts, metadatas=metadatas, text_ids=text_ids)
        else:
            # Difference vs default: omit split_mode; rely on semantic DB default split behavior
            self.semantic_db.insert_in_chunks(doc, metadata=metadata, max_sentences_per_chunk=max_sentences_per_chunk)


    def semantically_contextualize(self, message, semantic_top_k=1, semantic_where=None, semantic_contextualize_prompt=None, semantic_debug=False, semantic_context_max=1):
        # Retrieve candidates
        results = self.semantic_db.query(message, top_k=semantic_top_k, where=semantic_where)

        # Normalize message and compute intent flags for downstream exact-ID fallback
        normalized_msg, msg_tokens = normalize_query_text(message)
        wants_item, wants_pokemon, wants_move, _ = intent_from_tokens(msg_tokens)

        # Print only candidate "<category> — <id>" labels when debugging, to reduce noise
        if semantic_debug:
            print("Context candidate ids:", candidate_labels(results))

        # Choose a single best context by comparing the message to each entry's JSON 'id'
        # If parsing fails or no IDs are present, fall back to the first available text.
        best, best_score = select_best_by_id_similarity(results, normalized_msg, msg_tokens)

        # Final context string: single best text (not a join)
        context = None
        if best is not None:
            context = best.get("text") or ""

        # Exact-ID fallback: directly load the exact KB line if intent is clear
        # but semantic retrieval misses the target.
        #
        # Test examples (run with --debug):
        # 1) Query: "Information about the Item Pep-Up Plant"
        #    - Expected: after item-only filtering, if "Pep-Up Plant" isn't in top-k,
        #      this fallback finds the exact KB line starting with:
        #      "Item — Pep-Up Plant — ..." and sets chosen_id= Pep-Up Plant, score= 1.0.
        # 2) Query: "Pokemon Piplup"
        #    - Expected: filters to Pokémon; if the exact "Pokemon — Piplup — ..." line
        #      isn't retrieved, fallback loads it and chosen_id= Piplup, score= 1.0.
        # 3) Query: "move Ice Beam"
        #    - Expected: filters to moves; fallback loads "Move — Ice Beam — ..." when missing.
        try:
            wants_any = wants_item or wants_pokemon or wants_move  # e.g., query "Item Pep-Up Plant" → wants_item=True
            weak_match = (best_score < 0.98) or (context is None)  # fallback triggers if top-k missed the exact target
            if wants_any and weak_match:  # only do exact lookup when intent is clear and match is weak
                probable_id, _ = extract_probable_id(normalized_msg, msg_tokens)  # "Item Pep-Up Plant" → probable_id="pep-up plant"
                kb_header = kb_header_from_flags(wants_item, wants_pokemon, wants_move)  # maps to "item" | "pokemon" | "move"
                kb_path = resolve_kb_path(semantic_where)  # defaults to data/processed/pokemon_kb.txt when doc_name not set
                exact_line = find_exact_kb_line(kb_path, kb_header, probable_id)  # finds "Item — Pep-Up Plant — ..." if present
                if exact_line:  # when found, use exact KB line for context
                    context = exact_line  # debug chosen_id becomes "Pep-Up Plant"
                    best_score = 1.0  # force select this exact match
        except Exception:
            # Ignore fallback failures
            pass

        # Include semantic context if applicable
        if context is not None:
            if semantic_debug:
                # Debug: show chosen header id and similarity score
                parts = context.strip().split(" — ")  # e.g., "Item — Pep-Up Plant — ..." → ["Item", "Pep-Up Plant", ...]
                chosen_id = parts[1].strip() if len(parts) >= 2 else None  # Examples: "Pep-Up Plant" | "Piplup" | "Ice Beam"
                print("\nSemantic retrieval debug: chosen_id=", chosen_id, " score=", best_score)  # Expect score=1.0 when exact-ID fallback is used
                print("Context text:", context)  # Full KB line for verification, e.g., "Item — Pep-Up Plant — ..."
                
            if semantic_contextualize_prompt is None:
                self.add_context("This information may be relevant to the conversation ... " + str(context))
            else:
                self.add_context(str(semantic_contextualize_prompt) + " ... " + str(context))
        else:
            print("semantically_contextualize: No context added")

    def discuss_document(self, semantic_query, doc_name=None, semantic_where=None, semantic_top_k=5, semantic_contextualize_prompt=None, semantic_debug=False):
        # Matches base method’s API; prepares a where filter when doc_name is provided
        if semantic_where is None and doc_name is None:
            raise ValueError("Must include either doc_name or semantic_where")
        if semantic_where is None and doc_name is not None:
            semantic_where = {"doc_name": doc_name}
        self.semantically_contextualize(
            semantic_query,
            semantic_top_k=semantic_top_k,
            semantic_where=semantic_where,
            semantic_contextualize_prompt=semantic_contextualize_prompt,
            semantic_debug=semantic_debug,
        )

