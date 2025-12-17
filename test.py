from vendor_patches.patch_noahs_agent import PatchedAgent as ollama_chat_agent

#  Analyze a document without having to upload entire document into context window

# 1. create agent, ollama api must be running in background and provided model must be installed
agent = ollama_chat_agent(name="Test", model="llama3.2:3b")

# 2. (optional) purge semantic database for fresh start
agent.semantic_db.purge_collection();

# 3. upload consolidated Pokémon knowledge base split by lines so each entry is intact
agent.upload_document("data/processed/pokemon_kb.txt", max_sentences_per_chunk=1, split_mode="lines")

# 4. do a semantic search for 5 relevant passages from the consolidated KB and add them to the context window
semantic_query = input("Enter a semantic search query: ")
agent.discuss_document(semantic_query, doc_name="pokemon_kb.txt", semantic_top_k=10, semantic_debug=True)
# 5. start the chat with a question about the uploaded content.
message = input("Enter your question about the uploaded document: ")
agent.discuss_document(message, doc_name="pokemon_kb.txt", semantic_top_k=10, semantic_debug=True)
response_stream = agent.chat(message, show=False)
agent.print_stream(response_stream)



print("\n")
print("Continue to chat...")
print("\n")

while message not in ["bye","goodbye","exit","quit"]:
	message = input("You: ")
	# Refresh context from the KB based on the latest user message
	agent.discuss_document(message, doc_name="pokemon_kb.txt", semantic_top_k=10, semantic_debug=True)
	response_stream = agent.chat(message, show=False)
	agent.print_stream(response_stream)
