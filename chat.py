from vendor_patches.patch_noahs_agent import PatchedAgent as ollama_chat_agent
import argparse

# 1. create agent, ollama api must be running in background and provided model must be installed
parser = argparse.ArgumentParser(description="Cynthia chat with debug")
parser.add_argument("--debug", action="store_true", help="Enable semantic retrieval debug output")
args = parser.parse_args()

agent = ollama_chat_agent(name="Cynthia", model="llama3.2:3b")
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
	# Simple streaming: print chunks directly
	response_stream = agent.chat(message, show=False)
	agent.print_stream(response_stream)
