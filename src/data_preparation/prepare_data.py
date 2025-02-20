import re
import json
import os

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Divide o texto em blocos de `chunk_size`.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start: start + chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap  
    return chunks

def save_chunks(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def main():
    input_filepath = "../../data/raw/the Origin of Species.txt" 
    output_filepath = "../../data/processed/chunks.json"
    
    raw_text = load_text(input_filepath)
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    print(f"Total de chunks gerados: {len(chunks)}")
    
    save_chunks(chunks, output_filepath)

if __name__ == '__main__':
    main()
