# from qa.rag_chain import build_qa_chain

# qa_chain = build_qa_chain()
# query = input("Ask me anything: ")
# result = qa_chain.invoke({"question": query})
# print(result)
# ui/app.py

# import gradio as gr
# from qa.rag_chain import build_qa_chain

# # Build the RetrievalQA chain
# qa_chain = build_qa_chain()

# # Define the chatbot function
# def chatbot(query):
#     result = qa_chain.invoke({"query": query})
#     return result["result"]  # or result['answer'] depending on your output keys

# # Create Gradio interface
# iface = gr.Interface(
#     fn=chatbot,
#     inputs=gr.Textbox(lines=2, placeholder="Ask me anything..."),
#     outputs=gr.Textbox(),
#     title="📘 RAG Chatbot",
#     description="Ask questions and get answers based on document context."
# )

# # Launch the interface
# if __name__ == "__main__":
#     iface.launch(share=True)

import gradio as gr
from qa.rag_chain import build_qa_chain

# Build the RAG RetrievalQA chain
qa_chain = build_qa_chain()

# Define the response logic
def respond(message, history):
    # Optional: Detect Arabic (for multilingual support)
    lang = "ar" if any("\u0600" <= c <= "\u06FF" for c in message) else "en"

    try:
        # Query the RAG chain — context is retrieved automatically from vectorstore
        response = qa_chain.invoke({"query": message})
        answer = response.get("result", "").strip()

        # Debugging: Uncomment to view retrieved context
        for i, doc in enumerate(response.get("source_documents", [])):
            print(f"\nDocument {i+1}:\n", doc.page_content[:300])

        # Post-processing logic
        if "not relevant" in answer.lower():
            return "This is not relevant to the question"
        if "don't know" in answer.lower():
            return "I don't know"
        if len(answer.strip()) == 0:
            return "I don't know"

        return answer

    except Exception as e:
        print("Error:", str(e))
        return "Sorry, an error occurred."

# Gradio chat interface
chatbot_ui = gr.ChatInterface(
    fn=respond,
    title="📘 RAG Legal Assistant",
    description="Ask your question and get a concise answer based on your documents.",
    chatbot=gr.Chatbot(),
    textbox=gr.Textbox(placeholder="Ask your question...", container=False),
)

# Launch the Gradio app
if __name__ == "__main__":
    chatbot_ui.launch(share=True)
