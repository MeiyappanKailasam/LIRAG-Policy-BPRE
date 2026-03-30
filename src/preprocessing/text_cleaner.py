import re
def clean_text(text):
    text=re.sub(r'\n+','\n',text)
    text=re.sub(r'Page\s+\d+','',text)
    text=re.sub(r'\s{2+}','',text)
    return text.strip()
if __name__ == "__main__":
    with open("data/processed_clauses/raw_text.txt","r",encoding="utf-8") as f:
        raw=f.read()
    cleaned=clean_text(raw)
    with open("data/processed_clauses/cleaned_text.txt","w",encoding="utf-8") as f:
        f.write(cleaned)
