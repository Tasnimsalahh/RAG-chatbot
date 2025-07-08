from langchain.text_splitter import RecursiveCharacterTextSplitter
from loaders.pdf_loaders import extract_text_from_pdf

def chunk_text(file_path, chunk_size=500, chunk_overlap=200):
    """
    Load text from a file and split it into chunks.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of text chunks.
    """
    text = extract_text_from_pdf(file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return text_splitter.split_text(text)

if __name__ == "__main__":
    file_path = "docs/Iranian attack on Israel.pdf"
    chunks = chunk_text(file_path)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")