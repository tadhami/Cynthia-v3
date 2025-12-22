from noahs_local_ollama_chat_agent.agent import ollama_chat_agent as BaseAgent
from noahs_local_ollama_chat_agent.local_sql_db import local_sql_db
from noahs_local_ollama_chat_agent.local_semantic_db import local_semantic_db
import os
import json
from difflib import SequenceMatcher


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
            self.semantic_db.batch_insert(texts=texts, metadatas=metadatas, text_ids=text_ids)
        else:
            # Difference vs default: omit split_mode; rely on semantic DB default split behavior
            self.semantic_db.insert_in_chunks(doc, metadata=metadata, max_sentences_per_chunk=max_sentences_per_chunk)


    def semantically_contextualize(self, message, semantic_top_k=1, semantic_where=None, semantic_contextualize_prompt=None, semantic_debug=False, semantic_context_max=1):
        # Retrieve candidates
        results = self.semantic_db.query(message, top_k=semantic_top_k, where=semantic_where)
        # Print only candidate "<category> — <id>" labels when debugging, to reduce noise
        if semantic_debug:
            candidate_labels = []
            for item in results or []:
                text = item.get("text") or ""
                parts = text.strip().split(" — ")
                if len(parts) >= 2:
                    candidate_labels.append(f"{parts[0].strip()} — {parts[1].strip()}")
            print("Context candidate ids:", candidate_labels)

        # Choose a single best context by comparing the message to each entry's JSON 'id'
        # If parsing fails or no IDs are present, fall back to the first available text.
        best = None
        best_score = -1.0
        normalized_msg = str(message or "").strip().lower()
        msg_tokens = [t for t in normalized_msg.split() if t]

        for item in results or []:
            text = item.get("text") or ""
            parts = text.strip().split(" — ")
            candidate_id = (parts[1].strip().lower() if len(parts) >= 2 else "")

            # Compute similarity between the message and the candidate 'id' only
            score = 0.0
            if candidate_id:
                # Exact match
                if candidate_id == normalized_msg:
                    score = 1.0
                # Substring match
                elif candidate_id in normalized_msg:
                    score = 0.99
                # Token intersection heuristic
                elif any(tok in msg_tokens for tok in candidate_id.split()):
                    score = 0.95
                else:
                    # Partial ratio: best against full message or any individual token
                    score = SequenceMatcher(None, normalized_msg, candidate_id).ratio()
                    for tok in msg_tokens:
                        score = max(score, SequenceMatcher(None, tok, candidate_id).ratio())

            # Prefer higher ID-based score; tie-breaker uses semantic distance if available
            if score > best_score:
                best_score = score
                best = item
            elif score == best_score and best is not None:
                # Tie-break on smaller distance (closer semantic match)
                try:
                    d_cur = float(item.get("distance")) if item.get("distance") is not None else float("inf")
                    d_best = float(best.get("distance")) if best.get("distance") is not None else float("inf")
                    if d_cur < d_best:
                        best = item
                except Exception:
                    pass

        # Fallback if no score improved or list empty
        if best is None and results:
            best = results[0]

        # Final context string: single best text (not a join)
        context = None
        if best is not None:
            context = best.get("text") or ""

        # Include semantic context if applicable
        if context is not None:
            if semantic_debug:
                # Debug: show chosen header id and similarity score
                parts = context.strip().split(" — ")
                chosen_id = parts[1].strip() if len(parts) >= 2 else None
                print("\nSemantic retrieval debug: chosen_id=", chosen_id, " score=", best_score)
                print("Context text:", context)
                
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

