import os
import pandas as pd

INPUT_CSV = "data/raw_policies/updated_data.csv"
OUTPUT_DIR = "data/processed_clauses"
MAX_SCHEMES = 100 # Adjust this if you want to index all 3400!

def csv_to_text():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Take first MAX_SCHEMES
    df_subset = df.head(MAX_SCHEMES)
    
    for idx, row in df_subset.iterrows():
        name = str(row.get('scheme_name', f'Scheme_{idx}')).strip()
        
        # Create a valid filename
        safe_name = "".join([c if c.isalnum() else "_" for c in name])
        if not safe_name:
            safe_name = f"scheme_{idx}"
        
        filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"SCHEME NAME: {name}\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("DETAILS:\n")
            f.write(str(row.get('details', '')) + "\n\n")
            
            f.write("BENEFITS:\n")
            f.write(str(row.get('benefits', '')) + "\n\n")
            
            f.write("ELIGIBILITY:\n")
            f.write(str(row.get('eligibility', '')) + "\n\n")
            
            f.write("APPLICATION PROCESS:\n")
            f.write(str(row.get('application', '')) + "\n\n")
            
            f.write("DOCUMENTS REQUIRED:\n")
            f.write(str(row.get('documents', '')) + "\n\n")
            
            f.write("TAGS:\n")
            f.write(str(row.get('tags', '')) + "\n\n")
            
    print(f"Successfully generated {len(df_subset)} policy text files in {OUTPUT_DIR}")

if __name__ == "__main__":
    csv_to_text()
