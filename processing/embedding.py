import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from processing.chunking import chunk_text
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DB_DIR = "chroma_db"

# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
embedding_function = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
)

def embed_text(file_path):
    chunks = chunk_text(file_path, chunk_size=500, chunk_overlap=200)
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Store in Chroma
    db = Chroma.from_documents(documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)
    db.persist()
    return documents

if __name__ == "__main__":
    file_path = "docs/Iranian attack on Israel.pdf"
    embeddings = embed_text(file_path)