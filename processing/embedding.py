import os
import shutil
import torch
import json
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from processing.data_preparation import chunk_text

os.environ["TRANSFORMERS_NO_TF"] = "1"

CHROMA_DB_DIR = "chroma_db"
JSON_FILE = "docs/processed_document.json"

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
)

def embed_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        sections_data = json.load(f)

    documents = []
    for item in sections_data:
        filename = item.get("filename", "unknown")
        for section in item.get("sections", []):
            title = section.get("title", "Untitled")
            content = section.get("content", "").strip()
            keywords = section.get("keywords", [])
            keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)

            doc = Document(
                page_content=content,
                metadata={
                    "title": title,
                    "filename": filename,
                    "keywords": keywords_str
                }
            )
            documents.append(doc)

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    # Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print(f"✅ Embedded and stored {len(documents)} sections in ChromaDB.")

def embed_text(file_path):
    chunks = chunk_text(file_path, chunk_size=500, chunk_overlap=200)
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Store in Chroma
    db = Chroma.from_documents(documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)
    db.persist()
    return documents

if __name__ == "__main__":
    embed_from_json(JSON_FILE)