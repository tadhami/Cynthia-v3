from noahs_local_ollama_chat_agent.agent import ollama_chat_agent as BaseAgent
from noahs_local_ollama_chat_agent.local_sql_db import local_sql_db
from .patched_semantic_db import PatchedSemanticDB
import os
import json
import requests

class PatchedAgent(BaseAgent):
    def __init__(self, model="llama3.2:3b", model_encoding=None, name="Agent", url="http://localhost:11434/api", context_window_limit=2048, semantic_model_name="multi-qa-MiniLM-L6-cos-v1"):
        # Store semantic model choice, then run base init
        self.semantic_model_name = semantic_model_name
        super().__init__(model=model, model_encoding=model_encoding, name=name, url=url, context_window_limit=context_window_limit)
    def _initialize_databases(self):
        """Initialize DBs but use patched semantic DB."""
        self.semantic_db = PatchedSemanticDB(
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
        self.semantic_db.insert_in_chunks(doc, metadata=metadata, max_sentences_per_chunk=max_sentences_per_chunk, split_mode=split_mode)

    def semantically_contextualize(self, message, semantic_top_k=1, semantic_where=None, semantic_contextualize_prompt=None, semantic_debug=False, semantic_context_max=1):
        # Retrieve candidates semantically
        context_results = self.semantic_db.query(message, top_k=semantic_top_k, where=semantic_where)
        # Limit injected context to top-N (default 1) to avoid blending across similar entries
        selected = context_results[:max(1, int(semantic_context_max))]
        context_text = [d.get("text", "") for d in selected]
        context = " ... ".join(context_text) if context_text else None

        if context:
            print("semantically_contextualize: context added")
            if semantic_debug:
                print("\nSemantic retrieval debug:")
                for item in selected:
                    meta = item.get("metadata") or {}
                    excerpt = (item.get("text") or "")[:240].replace("\n", " ")
                    doc_name = meta.get("doc_name")
                    index = meta.get("index")
                    print(f"- id: {item.get('id')} | distance: {item.get('distance')}")
                    if doc_name is not None:
                        print(f"  doc_name: {doc_name}")
                    if index is not None:
                        print(f"  index: {index}")
                    print(f"  excerpt: {excerpt}")
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

    def chat(self, message, show=False, stream=True, conversation=True, auto_refresh=True,
             show_tokens_left=False, refresh_summary_size=500, refresh_summarize_prompt=None,
             refresh_summary_reference_prompt=None, speech_ready=False):
        """
        Override to fix refresh: ensure the current user message is present after refresh.
        Also avoid counting the message twice when checking the context window.
        """
        # Pre-check context window BEFORE appending the user message to avoid double counting
        if auto_refresh and not self.is_within_context_window(current_message=message):
            if show:
                print(f"{self.name}: CONTEXT WINDOW LIMIT ABOUT TO GO OUT OF BOUNDS, REFRESHING CONVERSATION")
            self.refresh_conversation(summary_size=refresh_summary_size,
                                      summarize_prompt=refresh_summarize_prompt,
                                      summary_reference_prompt=refresh_summary_reference_prompt)

        # Now append the user message (so it's always present in the payload)
        if conversation:
            self.conversation_history.append({"role": "user", "content": message})

        if show_tokens_left and auto_refresh:
            print(f"{self.name}: Tokens left until refresh: {self.tokens_left()}")

        payload = {
            "model": self.model,
            "messages": self.conversation_history
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.url+"/chat", headers=headers, json=payload, stream=stream)

        if stream:
            def stream_generator():
                full_text = ""
                if show:
                    print(self.name + ": ", end="", flush=True)
                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:
                        try:
                            data = json.loads(chunk)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                if show:
                                    print(content, end="", flush=True)
                                full_text += content
                                yield content
                        except json.JSONDecodeError:
                            continue
                if conversation and full_text:
                    self.conversation_history.append({"role": "assistant", "content": full_text})
            return stream_generator()
        else:
            full_text = ""
            if show:
                print(self.name + ": ", end="")
            if response.status_code == 200:
                try:
                    json_objects = response.text.strip().split("\n")
                    for obj in json_objects:
                        data = json.loads(obj)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            if show:
                                print(content, end="", flush=True)
                            full_text += content
                except json.JSONDecodeError:
                    full_text = f"Error: Invalid JSON response from Ollama API"
            else:
                full_text = f"Error: {response.status_code}, {response.text}"
            if conversation and full_text:
                self.conversation_history.append({"role": "assistant", "content": full_text})
            return full_text
