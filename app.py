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

# Build the RAG chain
qa_chain = build_qa_chain()

# Static context to be used for every question (could be dynamic too)
CONTEXT = """
Iranian attack, along with the promise of a response, aims to restore a 
semblance of stability within Israel amidst internal turmoil.
Potential Scenarios in the Wake of the Iranian Attack
The unprecedented Iranian attack on targets within Israel has undoubt­edly 
established new parameters for engagement between the two adversaries, 
potentially reshaping future confrontations. The repercussions of this attack 
may unfold into various scenarios, influenced by...
"""

# Response logic with enforced behavior
def respond(message, history):
    # Optional: Detect language (you can improve this with langdetect)
    lang = "ar" if any("\u0600" <= c <= "\u06FF" for c in message) else "en"

    try:
        # Inject query and context
        response = qa_chain.invoke({
            "query": message,
            "context": CONTEXT
        })

        answer = response.get("result", "").strip()

        # Post-processing rules
        if "not found" in answer.lower():
            return "I don't know"
        if "not relevant" in answer.lower():
            return "This is not relevant to the question"
        if len(answer.strip()) == 0:
            return "I don't know"

        return answer

    except Exception:
        return "Sorry, an error occurred."

# Build the Gradio chatbot interface
chatbot_ui = gr.ChatInterface(
    fn=respond,
    title="📘 RAG Legal Assistant",
    description="Ask your question and get a brief answer based on the provided context.",
    chatbot=gr.Chatbot(),
    textbox=gr.Textbox(placeholder="Ask your question...", container=False),
)

# Run the app
if __name__ == "__main__":
    chatbot_ui.launch(share=True)
