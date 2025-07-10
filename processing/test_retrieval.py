from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the directory where the DB was persisted
CHROMA_DB_DIR = "chroma_db"

# Set up the same embedding function used during creation
embedding_function = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cuda"}
)

# Load the vectorstore
db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)

# Ask a test query
query = "What did Iran do to Israel?"

# Retrieve top 3 similar chunks
results = db.similarity_search(query, k=3)

# Print the results
for i, result in enumerate(results):
    print(f"\nResult {i + 1}:\n{result.page_content}")
