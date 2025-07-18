import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
import json
from processing.chunking import chunk_text
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

CHROMA_DB_DIR = "chroma_db"
JSON_FILE = "docs/processed_sections.json"

embedding_function = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
)
# def embed_from_json(json_path: str):
#     with open(json_path, "r", encoding="utf-8") as f:
#         sections = json.load(f)

#     documents = [
#         Document(
#             page_content=section["content"],
#             metadata={"title": section["title"], "keywords": section.get("keywords", [])}
#         )
#         for section in sections
#     ]
#     if os.path.exists(CHROMA_DB_DIR):
#         import shutil
#         shutil.rmtree(CHROMA_DB_DIR)

#     vectorstore = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_function,
#         persist_directory=CHROMA_DB_DIR
#     )
#     vectorstore.persist()
#     print(f"✅ Embedded and stored {len(documents)} sections in ChromaDB.")

def embed_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for file_entry in data:
        filename = file_entry.get("filename", "unknown")
        sections = file_entry.get("sections", [])

        for section in sections:
            content = section.get("content")
            if not content:
                print(f"⚠️ Skipping section without content in {filename}")
                continue

            # Convert keywords list to comma-separated string
            keywords = section.get("keywords", [])
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)

            documents.append(
                Document(
                    page_content=content,
                    metadata = filter_complex_metadata({
                        "title": section.get("title", ""),
                        "keywords": section.get("keywords", []),
                        "source": filename,
                    })
                )
            )

    if not documents:
        raise ValueError("No valid sections with content found in the JSON.")

    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        shutil.rmtree(CHROMA_DB_DIR)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print(f"✅ Embedded and stored {len(documents)} sections from {len(data)} files into ChromaDB.")


def embed_text(file_path):
    chunks = chunk_text(file_path, chunk_size=500, chunk_overlap=200)
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Store in Chroma
    db = Chroma.from_documents(documents, embedding=embedding_function, persist_directory=CHROMA_DB_DIR)
    db.persist()
    return documents

if __name__ == "__main__":
    embed_from_json(JSON_FILE)