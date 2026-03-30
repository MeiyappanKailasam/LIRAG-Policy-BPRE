import re
import json
import os
import argparse

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_FILE = "data/clauses.json"

# Regex to capture clause numbers like:
# 1. , 1.1 , 5.1 , 11. , 13.4 etc.
CLAUSE_PATTERN = re.compile(r"\n\s*(\d+(\.\d+)*)\s+")

# -----------------------------
# CORE FUNCTIONS
# -----------------------------

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def segment_into_clauses(text, policy_id):
    """
    Splits raw policy text into numbered clauses.
    Returns a list of clause dicts.
    """
    clauses = []

    matches = list(CLAUSE_PATTERN.finditer(text))

    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        clause_id = matches[i].group(1)
        clause_text = text[start:end].strip().replace("\n", " ")

        if len(clause_text) < 20:
            continue  # skip very small/noisy clauses

        clause = {
            "policy_id": policy_id,
            "clause_id": clause_id,
            "text": clause_text
        }
        clauses.append(clause)

    return clauses

def append_clauses(new_clauses):
    """
    Appends new clauses to clauses.json safely.
    """
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_clauses = json.load(f)
    else:
        existing_clauses = []

    existing_clauses.extend(new_clauses)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_clauses, f, indent=2, ensure_ascii=False)

# -----------------------------
# MAIN
# -----------------------------

def main(input_txt, policy_id):
    print(f"Processing policy: {policy_id}")
    print(f"Input file: {input_txt}")

    text = load_text(input_txt)
    clauses = segment_into_clauses(text, policy_id)

    print(f"Extracted {len(clauses)} clauses")

    append_clauses(clauses)

    print(f"Clauses appended to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert policy text to clauses")
    parser.add_argument("--input", required=True, help="Path to extracted policy text (.txt)")
    parser.add_argument("--policy_id", required=True, help="Unique policy identifier")

    args = parser.parse_args()

    main(args.input, args.policy_id)