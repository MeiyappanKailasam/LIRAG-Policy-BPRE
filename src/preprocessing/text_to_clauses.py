import re
import json
import os
import argparse
from tqdm import tqdm

OUTPUT_FILE = "data/clauses.json"
INPUT_DIR = "data/processed_clauses"

# Match numbers like "1.", "1.1", "5.1", etc.
CLAUSE_PATTERN = re.compile(r"\n\s*(\d+(\.\d+)*)\s+")
# Fallback match for headers like "DETAILS:", "BENEFITS:"
HEADER_PATTERN = re.compile(r"\n([A-Z\s]+):")

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def segment_into_clauses(text, policy_id):
    clauses = []
    
    # Try finding numbered clauses first
    matches = list(CLAUSE_PATTERN.finditer(text))
    
    # If no numbered clauses, try header-based chunking
    if not matches:
        matches = list(HEADER_PATTERN.finditer(text))
        
    # If still no matches, just split by double newline (paragraphs)
    if not matches:
        paragraphs = text.split("\n\n")
        for i, p in enumerate(paragraphs):
            if len(p.strip()) > 20:
                clauses.append({
                    "policy_id": policy_id,
                    "clause_id": str(i+1),
                    "text": p.strip().replace("\n", " ")
                })
        return clauses

    # Process regex matches
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        clause_id = matches[i].group(1).strip()
        clause_text = text[start:end].strip().replace("\n", " ")

        if len(clause_text) < 20:
            continue
            
        clauses.append({
            "policy_id": policy_id,
            "clause_id": clause_id,
            "text": clause_text
        })

    return clauses

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} not found.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    all_clauses = []
    
    print(f"Segmenting {len(files)} files...")
    
    for f in tqdm(files, desc="Processing text files"):
        policy_id = f.replace(".txt", "")
        filepath = os.path.join(INPUT_DIR, f)
        
        try:
            text = load_text(filepath)
            clauses = segment_into_clauses(text, policy_id)
            all_clauses.extend(clauses)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"Extracted a total of {len(all_clauses)} clauses.")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(all_clauses, out, indent=2, ensure_ascii=False)
        
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()