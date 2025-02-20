import os
import json
from time import sleep
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.embeddings.embedding_service import generate_and_save_embeddings

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("A variável de ambiente PINECONE_API_KEY não está definida.")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Definir o nome do índice e a dimensão dos embeddings
INDEX_NAME = "origin-of-species-index"
EMBEDDING_DIM = 1536 

indexes = pc.list_indexes().names()
if INDEX_NAME not in indexes:
    print(f"Criando o índice {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric='cosine',
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD"),
            region=os.getenv("PINECONE_ENV")
        )
    )
    sleep(10) 

index = pc.Index(INDEX_NAME)

chunks_filepath = "../../data/processed/chunks.json"
with open(chunks_filepath, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total de chunks carregados: {len(chunks)}")

embeddings_filepath = "../../data/embeddings/embeddings.json"

if not os.path.exists(embeddings_filepath):
    print("Embeddings não encontrados. Gerando embeddings e salvando em arquivo...")
    generate_and_save_embeddings(chunks, embeddings_filepath)

with open(embeddings_filepath, "r", encoding="utf-8") as f:
    embeddings = json.load(f)

print(f"Total de embeddings carregados: {len(embeddings)}")

vectors = []
for item in embeddings:
    vectors.append((item["id"], item["embedding"], {"text": item["text"]}))

batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i : i + batch_size]
    index.upsert(vectors=batch)
    print(f"Upsert dos vetores {i} a {i + len(batch) - 1} realizado.")

print("Processo de indexação concluído!")