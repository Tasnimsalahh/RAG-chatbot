def chunk_text(pages, chunk_size=500, chunk_overlap=200):
    chunks = []
    for page_text in pages:
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunks.append(page_text[start:end])
            start += chunk_size - chunk_overlap
    return chunks
