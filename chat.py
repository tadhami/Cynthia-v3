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

# 4. do a semantic search for 5 relevant passages to a semantic_query and add it to the context window of the agent
semantic_query = input("Enter a semantic search query: ")
agent.discuss_document(semantic_query, doc_name="pokemon_kb.txt", semantic_top_k=5, semantic_debug=debug)

# 5. start the chat with a question about the uploaded content.
message = input("Enter your question about the uploaded document: ")
response_stream = agent.chat(message)
print("\n")
print(f"You: {message}")
try:
	first_chunk = next(response_stream)
except StopIteration:
	first_chunk = None

# If the first attempt yielded no chunks, retry once
if first_chunk is None:
	response_stream = agent.chat(message)
	try:
		first_chunk = next(response_stream)
	except StopIteration:
		first_chunk = None

if first_chunk is not None:
	def _prepend_stream(head, tail):
		yield head
		for c in tail:
			yield c
	agent.print_stream(_prepend_stream(first_chunk, response_stream))
else:
	# Still empty; print a newline to keep UX tidy
	print("")

print("\n")
print("Continue to chat...")
print("\n")

while message not in ["bye","goodbye","exit","quit"]:
	message = input("You: ")
	response_stream = agent.chat(message)
	agent.print_stream(response_stream)
