import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

openai.api_key = OPENAI_API_KEY

def get_embedding(text: str) -> list:
    """
    Gera o embedding para o texto fornecido.
    
    Parâmetros:
      text: O texto para o qual será gerado o embedding.
    
    Retorna:
      Uma lista contendo o vetor de embedding.
    """
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    return embedding

def generate_embeddings(chunks: list) -> list:
    """
    Gera embeddings para uma lista de chunks de texto.
    
    Parâmetros:
      chunks: Lista de textos (chunks) para os quais os embeddings serão gerados.
      
    Retorna:
      Uma lista de dicionários, cada um contendo:
        - "id": o índice do chunk (como string)
        - "embedding": o vetor de embedding gerado
        - "text": o texto do chunk
    """
    embeddings_list = []
    for idx, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            embeddings_list.append({
                "id": str(idx),
                "embedding": embedding,
                "text": chunk
            })
            print(f"Chunk {idx} processado com sucesso.")
        except Exception as e:
            print(f"Erro ao processar o chunk {idx}: {e}")
    return embeddings_list

def save_embeddings_to_json(embeddings: list, output_filepath: str):
    """
    Salva os embeddings gerados em um arquivo JSON.
    
    Parâmetros:
      embeddings: Lista de dicionários com embeddings.
      output_filepath: Caminho para o arquivo de saída.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    print(f"Embeddings salvos em {output_filepath}")

def generate_and_save_embeddings(chunks: list, output_filepath: str):
    """
    Gera embeddings para os chunks fornecidos e os salva em um arquivo JSON.
    
    Parâmetros:
      chunks: Lista de chunks de texto.
      output_filepath: Caminho do arquivo de saída para salvar os embeddings.
    """
    embeddings = generate_embeddings(chunks)
    save_embeddings_to_json(embeddings, output_filepath)
