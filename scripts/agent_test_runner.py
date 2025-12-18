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
    """Return ~50 test cases covering locations, evolutions, types/stats, items, and moves.
    Each test case is a dict with keys: semantic_query, question.
    """
    tests = []

    # Locations in specific games
    locations = [
        ("Regirock locations in Emerald", "Where can I find Regirock in Emerald?"),
        ("Registeel locations in Emerald", "Where can I find Registeel in Emerald?"),
        ("Abra locations in Red/Blue", "Where is Abra in Red and Blue?"),
        ("Aipom locations in Gold/Silver", "Where can I find Aipom in Gold or Silver?"),
        ("Aggron locations in Ruby/Sapphire/Emerald", "Where do I get Aggron in Ruby, Sapphire, or Emerald?"),
        ("Aerodactyl Old Amber revival", "How do I obtain Aerodactyl in FireRed/LeafGreen?"),
        ("Altaria locations in Ruby/Sapphire", "Where is Altaria in Ruby and Sapphire?"),
        ("Alcremie locations in Sword/Shield", "Where do I find Alcremie in Sword or Shield?"),
        ("Appletun raid locations", "Where can I find Appletun in Shield raids?"),
        ("Applin locations in Sword/Shield", "Where is Applin in the Galar region?"),
    ]

    # Evolutions
    evolutions = [
        ("Abra evolves", "How does Abra evolve?"),
        ("Kadabra to Alakazam evolution", "How does Kadabra evolve into Alakazam?"),
        ("Applin branching evolutions", "How does Applin evolve into Flapple and Appletun?"),
        ("Aegislash evolution", "How do you evolve Doublade into Aegislash?"),
        ("Amaura to Aurorus evolution", "How does Amaura evolve into Aurorus?"),
        ("Ambipom evolution requirement", "How do I evolve Aipom into Ambipom?"),
        ("Armaldo evolution", "How do you get Armaldo from Anorith?"),
        ("Azumarill evolution path", "How does Marill evolve into Azumarill?"),
        ("Avalugg evolution", "How does Bergmite evolve into Avalugg?"),
        ("Alolan forms evolution info", "Is Alolan Raichu an evolution and how?"),
    ]

    # Types, weaknesses, resistances, immunities
    typings = [
        ("Abomasnow weaknesses and resists", "What is Abomasnow weak to and what does it resist?"),
        ("Aegislash immunities", "What types is Aegislash immune to?"),
        ("Aggron resistances", "What resistances does Aggron have?"),
        ("Altaria weaknesses", "What is Altaria weak to?"),
        ("Arcanine resistances", "What types does Arcanine resist?"),
        ("Appletun weaknesses", "List Appletun's weaknesses and immunities."),
        ("Applin resistances", "What does Applin resist?"),
        ("Araquanid weaknesses", "What is Araquanid weak to?"),
        ("Arbok immunities", "Does Arbok have any immunities?"),
        ("Articuno typing and weaknesses", "What type is Articuno and its weaknesses?"),
    ]

    # Stats and metadata (classification, abilities, height/weight, base total)
    stats_meta = [
        ("Abra stats and classification", "What is Abra's classification, abilities, height/weight, and base total?"),
        ("Absol classification and abilities", "What is Absol's classification and abilities?"),
        ("Aggron base total and abilities", "What is Aggron's base stat total and abilities?"),
        ("Alakazam base total", "What is Alakazam's base stat total?"),
        ("Aromatisse classification", "What is Aromatisse's classification?"),
        ("Aron abilities", "What are Aron's abilities?"),
        ("Azumarill base total", "What is Azumarill's base total?"),
        ("Abomasnow height and weight", "What are Abomasnow's height and weight?"),
        ("Alcremie abilities", "What are Alcremie's abilities?"),
        ("Arcanine classification and stats", "Give Arcanine's classification and base stat total."),
    ]

    # Item locations across games
    items = [
        ("Master Ball locations", "Where can I find a Master Ball in different games?"),
        ("Exp. Share locations", "Where is the Exp. Share found across mainline games?"),
        ("Fire Stone locations", "Where can I get a Fire Stone?"),
        ("Water Stone locations", "Where can I get a Water Stone?"),
        ("Thunder Stone locations", "Where can I get a Thunder Stone?"),
        ("Leaf Stone locations", "Where can I get a Leaf Stone?"),
        ("Dusk Stone locations", "Where is Dusk Stone found?"),
        ("Soothe Bell locations", "Where is Soothe Bell located in different games?"),
        ("Linking Cord locations", "Where do I find a Linking Cord?"),
        ("Soul Dew locations", "Where can I get a Soul Dew?"),
    ]

    # Move questions (type, category, power, accuracy, PP, effect)
    moves = [
        ("Hyper Beam move details", "What are Hyper Beam's type, power, accuracy, PP, and effect?"),
        ("Thunderbolt move details", "Give Thunderbolt's power, accuracy, PP, and effect."),
        ("Earthquake move details", "What is Earthquake's power and effect?"),
        ("Ice Beam move details", "Summarize Ice Beam's type and effect."),
        ("Flamethrower move details", "What are Flamethrower's stats and effect?"),
        ("Surf move details", "Provide Surf's type, power, accuracy, PP, and effect."),
        ("Psychic move details", "What does the move Psychic do?"),
        ("Shadow Ball move details", "Summarize Shadow Ball's type and effect."),
        ("Stone Edge move details", "Give Stone Edge's power and accuracy."),
        ("Focus Blast move details", "What are Focus Blast's stats and effect?"),
    ]

    for category in (locations, evolutions, typings, stats_meta, items, moves):
        for sem, q in category:
            tests.append({"semantic_query": sem, "question": q})

    # Add a few more mixed cases to approach ~50
    extras = [
        ("Where to find Bulbasaur", "Where can I find Bulbasaur in early games?"),
        ("Beedrill typing and weaknesses", "What type is Beedrill and its weaknesses?"),
        ("Charizard move details", "What are notable moves for Charizard?"),
        ("Blastoise locations", "Where is Blastoise available across games?"),
        ("Butterfree moves", "List key move details for Butterfree."),
    ]
    for sem, q in extras:
        tests.append({"semantic_query": sem, "question": q})

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
