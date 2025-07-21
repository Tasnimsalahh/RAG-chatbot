# def chunk_text(pages, chunk_size=500, chunk_overlap=200):
#     chunks = []
#     for page_text in pages:
#         start = 0
#         while start < len(page_text):
#             end = start + chunk_size
#             chunks.append(page_text[start:end])
#             start += chunk_size - chunk_overlap
#     return chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(pages, chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    all_chunks = []
    for page in pages:
        chunks = text_splitter.split_text(page)
        all_chunks.extend(chunks)
    return all_chunks
        