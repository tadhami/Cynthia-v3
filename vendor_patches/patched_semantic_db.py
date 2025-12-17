from noahs_local_ollama_chat_agent.local_semantic_db import local_semantic_db as BaseDB

class PatchedSemanticDB(BaseDB):
    def purge_collection(self, batch_size=1000):
        """Delete all entries in batches to avoid SQL variable limits."""
        if self.collection is None:
            raise ValueError("No collection is currently set.")
        results = self.collection.get()
        if not results or not results.get("ids"):
            print(f"No entries found in collection: {self.collection_name}")
            return
        all_ids = results["ids"]
        total = len(all_ids)
        for i in range(0, total, batch_size):
            batch = all_ids[i:i+batch_size]
            self.collection.delete(ids=batch)
        print(f"Purged all {total} entries from collection: {self.collection_name}")

    def insert_in_chunks(self, text, metadata=None, text_id=None, max_sentences_per_chunk=42, split_mode="sentences"):
        """Insert text in chunks; supports 'lines' and 'sentences' split modes."""
        if split_mode == "lines":
            lines = text.splitlines()
            texts = []
            metadatas = []
            text_ids = []
            for i in range(0, len(lines), max_sentences_per_chunk):
                chunk_lines = lines[i:i + max_sentences_per_chunk]
                chunk = "\n".join(chunk_lines)
                texts.append(chunk)
                if metadata is not None:
                    m = eval(repr(metadata))
                    m["index"] = len(metadatas)
                    metadatas.append(m)
                if text_id is not None:
                    t = f"{text_id}-{len(text_ids)}"
                    text_ids.append(t)
            if metadatas == []:
                metadatas = None
            if text_ids == []:
                text_ids = None
            return self.batch_insert(texts=texts, metadatas=metadatas, text_ids=text_ids)
        # Fallback to base implementation for sentence splitting
        return super().insert_in_chunks(text, metadata=metadata, text_id=text_id, max_sentences_per_chunk=max_sentences_per_chunk)
