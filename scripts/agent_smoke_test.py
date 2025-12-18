import csv
import os
import sys
from pathlib import Path
import requests

# Ensure repo root is on sys.path for local imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vendor_patches.patch_noahs_agent import PatchedAgent as ollama_chat_agent

KB_PATH = ROOT / "data" / "processed" / "pokemon_kb.txt"
REPORTS_DIR = ROOT / "reports"
OUT_CSV = REPORTS_DIR / "agent_smoke_results.csv"


def build_test_cases():
    """Return 2 simple test cases to validate the runner logic."""
    return [
        {  # Game-specific location question present in KB
            "semantic_query": "Abra locations in Red/Blue",
            "question": "Where is Abra in Red and Blue?",
        },
        {  # Move details question present in KB
            "semantic_query": "Thunderbolt move details",
            "question": "Give Thunderbolt's power, accuracy, PP, and effect.",
        },
    ]


def main():
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {KB_PATH}")

    # Initialize agent
    agent = ollama_chat_agent(name="Cynthia", model="llama3.2:3b")

    # Check Ollama availability (optional, for graceful failure)
    ollama_ok = False
    try:
        resp = requests.get(agent.url + "/tags", timeout=3)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    # Upload KB split by lines so each entry is intact
    try:
        agent.upload_document(str(KB_PATH), max_sentences_per_chunk=1, split_mode="lines")
    except Exception as e:
        print(f"Error uploading KB: {e}")

    test_cases = build_test_cases()
    rows = []

    for idx, case in enumerate(test_cases, start=1):
        # Reset conversation to avoid cross-test bleed
        agent.conversation_history = []
        agent.add_context("Your name is " + agent.name)

        semantic_query = case["semantic_query"]
        question = case["question"]

        # Inject semantic context first, then the actual question
        try:
            agent.discuss_document(semantic_query, doc_name=KB_PATH.name, semantic_top_k=10, semantic_debug=False)
            agent.discuss_document(question, doc_name=KB_PATH.name, semantic_top_k=10, semantic_debug=False)
        except Exception as e:
            print(f"Error during semantic contextualization (test {idx}): {e}")

        # Get Cynthia's response (non-streaming for capture)
        response_text = ""
        try:
            if ollama_ok:
                response_text = agent.chat(question, show=False, stream=False)
            else:
                response_text = "ERROR: Ollama API unavailable; response not generated"
        except Exception as e:
            response_text = f"ERROR: chat failed: {e}"

        # Sanitize response to keep CSV rows tidy (no embedded newlines)
        if isinstance(response_text, str):
            safe_response = response_text.replace("\r", " ").replace("\n", " ").strip()
        else:
            safe_response = str(response_text)

        rows.append({
            "test_id": idx,
            "semantic_query": semantic_query.replace("\r", " ").replace("\n", " ").strip(),
            "question": question.replace("\r", " ").replace("\n", " ").strip(),
            "response": safe_response,
        })

    # Write CSV report
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_id", "semantic_query", "question", "response"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV.relative_to(ROOT)} with {len(rows)} smoke tests.")


if __name__ == "__main__":
    main()
