# from qa.rag_chain import build_qa_chain

# qa_chain = build_qa_chain()
# query = input("Ask me anything: ")
# result = qa_chain.invoke({"question": query})
# print(result)
# ui/app.py

import gradio as gr
from qa.rag_chain import build_qa_chain

# Build the RetrievalQA chain
qa_chain = build_qa_chain()

# Define the chatbot function
def chatbot(query):
    result = qa_chain.invoke({"query": query})
    return result["result"]  # or result['answer'] depending on your output keys

# Create Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything..."),
    outputs=gr.Textbox(),
    title="📘 RAG Chatbot",
    description="Ask questions and get answers based on document context."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)
