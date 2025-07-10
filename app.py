from qa.rag_chain import build_qa_chain

if __name__ == "__main__":
    qa_chain = build_qa_chain()
    
    question = "What was the scale of Iran’s attack on Israel?"
    result = qa_chain.invoke({"query": question})

    print("Answer:", result['result'])
    print("\nSource Documents:")
    for doc in result['source_documents']:
        print("-", doc.page_content[:200], "...\n")
