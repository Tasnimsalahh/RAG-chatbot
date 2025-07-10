from qa.rag_chain import build_qa_chain

if __name__ == "__main__":
    qa_chain = build_qa_chain()
    
    # question = "What was the scale of Iran’s attack on Israel?"
    # result = qa_chain.invoke({"query": question})

    # print("Answer:", result['result'])
    # print("\nSource Documents:")
    # for doc in result['source_documents']:
    #     print("-", doc.page_content[:200], "...\n")
    while True:
        query = input("Ask me anything: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.invoke({"query": query})
        print(response["result"])
        for i, doc in enumerate(response["source_documents"]):
            print(f"\nSource {i+1}:\n{doc.page_content[:300]}...")

