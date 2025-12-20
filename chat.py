from vendor_patches.patch_noahs_agent import PatchedAgent as ollama_chat_agent
from noahs_tts import TTS
import argparse

# 1. create agent, ollama api must be running in background and provided model must be installed
parser = argparse.ArgumentParser(description="Cynthia chat with optional TTS and debug")
parser.add_argument("--tts", action="store_true", help="Speak responses using text-to-speech")
parser.add_argument("--debug", action="store_true", help="Enable semantic retrieval debug output")
args = parser.parse_args()

agent = ollama_chat_agent(name="Cynthia", model="llama3.2:3b")
tts = TTS(voice="Samantha") if args.tts else None
debug = bool(args.debug)
# 2. (optional) purge semantic database for fresh start
agent.semantic_db.purge_collection();

# 3. upload consolidated Pokémon knowledge base split by lines so each entry is intact
agent.upload_document("data/processed/pokemon_kb.txt", max_sentences_per_chunk=1, split_mode="lines")

print("\n")
print("Continue to chat...")
print("\n")

message = ""
while message not in ["bye","goodbye","exit","quit"]:
	semantic_query = input("Enter a semantic search query: ")
	agent.discuss_document(semantic_query, doc_name="pokemon_kb.txt", semantic_top_k=10, semantic_debug=debug)
	message = input("Enter your question about the uploaded document: ")
	agent.discuss_document(message, doc_name="pokemon_kb.txt", semantic_top_k=10, semantic_debug=debug)
	if tts is not None:
		try:
			response_text = agent.chat(message, show=False, stream=False)
		except Exception as e:
			print(f"[diag] non-stream chat raised: {type(e).__name__}: {e}")
			response_text = ""
		if not response_text or not str(response_text).strip():
			print(f"[diag] non-stream empty; switching to streaming. msg_len={len(message)}, msg_excerpt='{message[:120]}'")
			# Fallback to streaming if non-streaming returns empty; capture chunks manually
			response_stream = agent.chat(message, show=False)
			chunks = []
			try:
				for chunk in response_stream:
					if isinstance(chunk, dict):
						for k in ("response", "message", "text", "content"):
							val = chunk.get(k)
							if val:
								chunks.append(str(val))
								break
						else:
							chunks.append(str(chunk))
					else:
						chunks.append(str(chunk))
			except Exception:
				pass
			response_text = "".join(chunks)
		print("Cynthia: " + (response_text or "[no response]"))
		try:
			tts.say(response_text)
		except Exception:
			pass
	else:
		response_stream = agent.chat(message, show=False)
		agent.print_stream(response_stream)
