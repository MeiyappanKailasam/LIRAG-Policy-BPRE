import re
import json
def segment_clauses(text):
    pattern=r'(\d+(\.\d+)+)\s+(.*)'
    lines=text.split('\n')
    clauses=[]
    current_clause=None
    for line in lines:
        match=re.match(pattern,line)
        if match:
            if current_clause:
                clauses.append(current_clause)
            current_clause={
                "clause_id":match.group(1),
                "text":match.group(3),
            }
        else:
            if current_clause:
                current_clause["text"]+=" "+line.strip()
    return clauses
if __name__ == "__main__":
    with open("data/processed_clauses/cleaned_text.txt","r",encoding="utf-8") as f:
        text=f.read()
    clauses=segment_clauses(text)
    with open("data/processed_clauses/clauses.json","w",encoding="utf-8") as f:
        json.dump(clauses,f,indent=2)