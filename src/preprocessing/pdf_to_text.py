from pypdf import PdfReader
import os
INPUT_DIR="data/raw_policies"
OUTPUT_DIR="data/processed_clauses"
os.makedirs(OUTPUT_DIR,exist_ok=True)
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"Warning: failed to extract page {page_num} from {os.path.basename(pdf_path)}: {e}")
            continue
        if text:
            full_text += text + "\n"
    return full_text
if __name__ == "__main__":
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, file)
            print(f"Processing: {file}")

            try:
                text = extract_text(pdf_path)
            except Exception as e:
                print(f"Warning: skipping {file} due to read error: {e}")
                continue

            output_file = file.replace(".pdf", ".txt")
            output_path = os.path.join(OUTPUT_DIR, output_file)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)