from noahs_local_ollama_chat_agent.agent import ollama_chat_agent as BaseAgent
from noahs_local_ollama_chat_agent.local_sql_db import local_sql_db
from noahs_local_ollama_chat_agent.local_semantic_db import local_semantic_db
import os
import json
import requests

class PatchedAgent(BaseAgent):
    def __init__(self, model="llama3.2:3b", model_encoding=None, name="Agent", url="http://localhost:11434/api", context_window_limit=2048, semantic_model_name="multi-qa-MiniLM-L6-cos-v1"):
        # Store semantic model choice, then run base init
        self.semantic_model_name = semantic_model_name
        super().__init__(model=model, model_encoding=model_encoding, name=name, url=url, context_window_limit=context_window_limit)
    def _initialize_databases(self):
        """Initialize DBs using the base semantic DB (minimal patch)."""
        self.semantic_db = local_semantic_db(
            persist_directory=f"{self.name}_data/{self.name}_semantic_db",
            collection_name=f"{self.name}_general",
            sentence_transformer_name=getattr(self, "semantic_model_name", None) or "multi-qa-MiniLM-L6-cos-v1",
        )
        self.sql_db = local_sql_db(f"{self.name}_data/{self.name}_sql_db")

    def upload_document(self, doc_path, doc_name=None, max_sentences_per_chunk=5, metadata=None, split_mode="sentences"):
        if doc_name is None:
            doc_name = os.path.basename(doc_path)
        with open(doc_path, "r") as file:
            doc = file.read()
        if metadata is None:
            metadata = {"doc_name": doc_name}
        if split_mode == "lines":
            # Pre-split by lines here to avoid patching the DB
            lines = doc.splitlines()
            texts, metadatas, text_ids = [], [], []
            idx = 0
            for i in range(0, len(lines), max_sentences_per_chunk):
                chunk = "\n".join(lines[i:i + max_sentences_per_chunk])
                texts.append(chunk)
                m = eval(repr(metadata))
                m["index"] = idx
                metadatas.append(m)
                text_ids.append(f"{doc_name}-{idx}")
                idx += 1
            self.semantic_db.batch_insert(texts=texts, metadatas=metadatas, text_ids=text_ids)
        else:
            # Use sentence-based chunking provided by the base DB
            self.semantic_db.insert_in_chunks(
                doc,
                metadata=metadata,
                max_sentences_per_chunk=max_sentences_per_chunk,
            )

    def semantically_contextualize(self, message, semantic_top_k=1, semantic_where=None, semantic_contextualize_prompt=None, semantic_debug=False, semantic_context_max=1):
        # Retrieve candidates semantically
        context_results = self.semantic_db.query(message, top_k=semantic_top_k, where=semantic_where)
        # Limit injected context to top-N (default 1) to avoid blending across similar entries
        selected = context_results[:max(1, int(semantic_context_max))]
        context_text = [d.get("text", "") for d in selected]
        context = " ... ".join(context_text) if context_text else None

        # Include semantic context if applicable
        if context is not None:
            print("semantically_contextualize: context added")
            if semantic_contextualize_prompt is None:
                self.add_context("This information may be relevant to the conversation ... " + str(context))
            else:
                self.add_context(str(semantic_contextualize_prompt) + " ... " + str(context))
        else:
            print("semantically_contextualize: No context added")

    def discuss_document(self, semantic_query, doc_name=None, semantic_where=None, semantic_top_k=5, semantic_contextualize_prompt=None, semantic_debug=False):
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

