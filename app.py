import gradio as gr
from qa.rag_chain import build_qa_chain, handle_greetings_and_thanks

# Build the RAG RetrievalQA chain
qa_chain = build_qa_chain()

# Define the response logic
def respond(message, history):
    greeting_thanks_response = handle_greetings_and_thanks(message)
    if greeting_thanks_response:
        return greeting_thanks_response

    try:
        response = qa_chain.invoke({"query": message})
        answer = response.get("result", "").strip()

        # Debugging output
        print("Question:", message)
        print("Answer:", answer)
        print("\nRetrieved Context:")
        for i, doc in enumerate(response.get("source_documents", [])):
            print(f"\nDocument {i+1}:\n", doc.page_content[:300])

        # Language check (majority detection instead of any char)
        def detect_lang(text):
            arabic_chars = sum("\u0600" <= c <= "\u06FF" for c in text)
            return "ar" if arabic_chars / max(len(text), 1) > 0.5 else "en"

        message_lang = detect_lang(message)
        answer_lang = detect_lang(answer)

        if message_lang != answer_lang:
            if message_lang == "ar":
                return "الإجابة غير متوفرة باللغة العربية، يرجى إعادة صياغة السؤال إن أمكن."
            else:
                return "The answer is not available in English. Please rephrase your question."
        return answer
    except Exception as e:
        print("Error:", str(e))
        return "Sorry, an error occurred."

# Gradio chat interface
chatbot_ui = gr.ChatInterface(
    fn=respond,
    title="📘 RAG Assistant",
    description="Ask your question and get a concise answer based on your documents.",
    chatbot=gr.Chatbot(),
    textbox=gr.Textbox(placeholder="Ask your question...", container=False),
)

# Launch the Gradio app
if __name__ == "__main__":
    chatbot_ui.launch(share=True)
