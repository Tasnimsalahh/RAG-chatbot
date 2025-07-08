import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
import langchain_community.vectorstores as Chroma
from sentence_transformers import SentenceTransformer
from processing.chunking import chunk_text

CHROMA_DB_DIR = "chroma_db"  # Define the directory for Chroma DB

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B" , device=device)

def embed_text(file_path):
    """
    Embed a list of texts using a pre-trained SentenceTransformer model.

    Args:
        texts (list): List of strings to embed.

    Returns:
        list: List of embeddings.
    """
    chunks = chunk_text(file_path, chunk_size=500, chunk_overlap=200)
    db = Chroma.from_documents(chunks, embedding=model, persist_directory=CHROMA_DB_DIR)
    db.persist()

    # return model.encode(chunks, convert_to_tensor=True)

if __name__ == "__main__":
    file_path = "docs\Iranian attack on Israel.pdf"
    embeddings = embed_text(file_path)
    for i, embedding in enumerate(embeddings):
        print(f"Embedding {i + 1}:\n{embedding}\n")