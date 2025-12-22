import csv
import os
import sys
from pathlib import Path
import requests

# Ensure repo root is on sys.path for local imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vendor_patches.patch_noahs_agent import PatchedAgent as ollama_chat_agent

# Paths
KB_PATH = ROOT / "data" / "processed" / "pokemon_kb.txt"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = REPORTS_DIR / "agent_test_results.csv"


def build_test_cases():
    """Return a broad set of test cases covering Pokemon, Moves, and Items.
    All semantic queries follow the convention: "Information about <Pokemon|Move|Item>".
    Each test case is a dict with keys: semantic_query, question.
    """
    tests = []

    # Pokemon: Locations and game-specific availability
    pokemon_locations = [
        ("Regirock", "Where can I find Regirock in Emerald?"),
        ("Registeel", "Where can I find Registeel in Emerald?"),
        ("Abra", "Where is Abra in Red and Blue?"),
        ("Aipom", "Where can I find Aipom in Gold or Silver?"),
        ("Aggron", "Where do I get Aggron in Ruby, Sapphire, or Emerald?"),
        ("Aerodactyl", "How do I obtain Aerodactyl in FireRed/LeafGreen?"),
        ("Altaria", "Where is Altaria in Ruby and Sapphire?"),
        ("Alcremie", "Where do I find Alcremie in Sword or Shield?"),
        ("Appletun", "Where can I find Appletun in Shield raids?"),
        ("Applin", "Where is Applin in the Galar region?"),
        ("Blastoise", "Where is Blastoise available across games?"),
        ("Bulbasaur", "Where can I find Bulbasaur in early games?"),
    ]

    # Pokemon: Evolutions
    pokemon_evolutions = [
        ("Abra", "How does Abra evolve?"),
        ("Kadabra", "How does Kadabra evolve into Alakazam?"),
        ("Applin", "How does Applin evolve into Flapple and Appletun?"),
        ("Doublade", "How do you evolve Doublade into Aegislash?"),
        ("Amaura", "How does Amaura evolve into Aurorus?"),
        ("Aipom", "How do I evolve Aipom into Ambipom?"),
        ("Anorith", "How do you get Armaldo from Anorith?"),
        ("Marill", "How does Marill evolve into Azumarill?"),
        ("Bergmite", "How does Bergmite evolve into Avalugg?"),
        ("Raichu", "Is Alolan Raichu an evolution and how?"),
    ]

    # Pokemon: Types, weaknesses, resistances, immunities
    pokemon_typings = [
        ("Abomasnow", "What is Abomasnow weak to and what does it resist?"),
        ("Aegislash", "What types is Aegislash immune to?"),
        ("Aggron", "What resistances does Aggron have?"),
        ("Altaria", "What is Altaria weak to?"),
        ("Arcanine", "What types does Arcanine resist?"),
        ("Appletun", "List Appletun's weaknesses and immunities."),
        ("Applin", "What does Applin resist?"),
        ("Araquanid", "What is Araquanid weak to?"),
        ("Arbok", "Does Arbok have any immunities?"),
        ("Articuno", "What type is Articuno and its weaknesses?"),
    ]

    # Pokemon: Stats and metadata (classification, abilities, height/weight, base total)
    pokemon_stats_meta = [
        ("Abra", "What is Abra's classification, abilities, height/weight, and base total?"),
        ("Absol", "What is Absol's classification and abilities?"),
        ("Aggron", "What is Aggron's base stat total and abilities?"),
        ("Alakazam", "What is Alakazam's base stat total?"),
        ("Aromatisse", "What is Aromatisse's classification?"),
        ("Aron", "What are Aron's abilities?"),
        ("Azumarill", "What is Azumarill's base total?"),
        ("Abomasnow", "What are Abomasnow's height and weight?"),
        ("Alcremie", "What are Alcremie's abilities?"),
        ("Arcanine", "Give Arcanine's classification and base stat total."),
    ]

    # Items: general locations and game-specific availability
    item_questions = [
        ("Master Ball", "Where can I find a Master Ball in different games?"),
        ("Exp. Share", "Where is the Exp. Share found across mainline games?"),
        ("Fire Stone", "Where can I get a Fire Stone?"),
        ("Water Stone", "Where can I get a Water Stone?"),
        ("Thunder Stone", "Where can I get a Thunder Stone?"),
        ("Leaf Stone", "Where can I get a Leaf Stone?"),
        ("Dusk Stone", "Where is Dusk Stone found?"),
        ("Soothe Bell", "Where is Soothe Bell located in different games?"),
        ("Linking Cord", "Where do I find a Linking Cord?"),
        ("Soul Dew", "Where can I get a Soul Dew?"),
        ("Honey", "Where do I obtain Honey?"),
        ("Honey", "Can I obtain Honey in Pokemon Diamond?"),
        ("Honey", "Which games can I obtain Honey?"),
    ]

    # Moves: type, category, stats, and effects
    move_questions = [
        ("Hyper Beam", "What are Hyper Beam's type, power, accuracy, PP, and effect?"),
        ("Thunderbolt", "Give Thunderbolt's power, accuracy, PP, and effect."),
        ("Earthquake", "What is Earthquake's power and effect?"),
        ("Ice Beam", "Summarize Ice Beam's type and effect."),
        ("Flamethrower", "What are Flamethrower's stats and effect?"),
        ("Surf", "Provide Surf's type, power, accuracy, PP, and effect."),
        ("Psychic", "What does the move Psychic do?"),
        ("Shadow Ball", "Summarize Shadow Ball's type and effect."),
        ("Stone Edge", "Give Stone Edge's power and accuracy."),
        ("Focus Blast", "What are Focus Blast's stats and effect?"),
    ]

    # Assemble tests using the "Information about …" semantic query convention
    for name, q in pokemon_locations:
        tests.append({"semantic_query": f"Information about {name}", "question": q})
    for name, q in pokemon_evolutions:
        tests.append({"semantic_query": f"Information about {name}", "question": q})
    for name, q in pokemon_typings:
        tests.append({"semantic_query": f"Information about {name}", "question": q})
    for name, q in pokemon_stats_meta:
        tests.append({"semantic_query": f"Information about {name}", "question": q})
    for name, q in item_questions:
        tests.append({"semantic_query": f"Information about {name}", "question": q})
    for name, q in move_questions:
        tests.append({"semantic_query": f"Information about {name}", "question": q})

    # A few more mixed Pokemon cases using the same convention
    extras = [
        ("Beedrill", "What type is Beedrill and its weaknesses?"),
        ("Charizard", "What are notable moves for Charizard?"),
        ("Butterfree", "List key move details for Butterfree."),
    ]
    for name, q in extras:
        tests.append({"semantic_query": f"Information about {name}", "question": q})

    return tests


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

        # Inject semantic context for the query and the actual question
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

        # Sanitize values to avoid CSV row breaks
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

    # Optionally send a graceful end message
    try:
        agent.chat("bye", show=False, stream=False)
    except Exception:
        pass

    # Write CSV report
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_id", "semantic_query", "question", "response"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV.relative_to(ROOT)} with {len(rows)} test cases.")


if __name__ == "__main__":
    main()
