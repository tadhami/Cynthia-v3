from vendor_patches.patch_noahs_agent import PatchedAgent as ollama_chat_agent
from noahs_tts import TTS
import argparse
#  Analyze a document without having to upload entire document into context window

# 1. create agent, ollama api must be running in background and provided model must be installed
parser = argparse.ArgumentParser(description="Cynthia chat with optional TTS")
parser.add_argument("--tts", action="store_true", help="Speak responses using text-to-speech")
args = parser.parse_args()

agent = ollama_chat_agent(name="Cynthia", model="llama3.2:3b")
tts = TTS(voice="Samantha") if args.tts else None
debug = True 
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
	# 5. start the chat with a question about the uploaded content.
	message = input("Enter your question about the uploaded document: ")
	agent.discuss_document(message, doc_name="pokemon_kb.txt", semantic_top_k=10, semantic_debug=debug)
	if tts is not None:
		# Use non-streaming to capture full text for TTS
		response_text = agent.chat(message, show=False, stream=False)
		print("Cynthia: " + response_text)
		try:
			tts.say(response_text)
		except Exception:
			pass
	else:
		response_stream = agent.chat(message, show=False)
		agent.print_stream(response_stream)
