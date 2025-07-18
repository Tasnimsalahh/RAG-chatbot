import gradio as gr
from qa.rag_chain import build_qa_chain, handle_greetings_and_thanks

# Build the RAG RetrievalQA chain
qa_chain = build_qa_chain()

# Define the response logic
def respond(message, history):
    greeting_thanks_response = handle_greetings_and_thanks(message)
    if greeting_thanks_response:
        return greeting_thanks_response
    
    # lang = "ar" if any("\u0600" <= c <= "\u06FF" for c in message) else "en"
    # if lang == "ar":
    #     system_prompt = "أنت مساعد قانوني ذكي. أجب على الأسئلة بناءً على السياق المقدم. إذا لم يكن لديك إجابة باللغة العربية، قل 'هذه اللغة غير مدعومة' وأجب باللغة الإنجليزية."
    # else:
    #     system_prompt = "You are a smart legal assistant. Answer questions based on the provided context. If you don't have an answer in Arabic, say 'this language is not supported' and answer in English."

    try:
        response = qa_chain.invoke({"query": message})
        answer = response.get("result", "").strip()

        # Debugging:
        print("Question:", message)
        print("Answer:", answer)
        print("\nRetrieved Context:")
        for i, doc in enumerate(response.get("source_documents", [])):
            print(f"\nDocument {i+1}:\n", doc.page_content[:300])

        # Ensure answer matches language
        lang = "ar" if any("\u0600" <= c <= "\u06FF" for c in message) else "en"
        if lang == "ar" and not any("\u0600" <= c <= "\u06FF" for c in answer):
            return "الإجابة غير متوفرة باللغة العربية، يرجى إعادة صياغة السؤال إن أمكن."
        elif lang == "en" and any("\u0600" <= c <= "\u06FF" for c in answer):
            return "The answer is not available in English. Please rephrase your question."

        return answer

        # # Extract the answer from the response
        # if "Answer:" in answer:
        #     extracted_answer = answer.split("Answer:")[-1].strip() if "Answer:" in answer else answer

        # return extracted_answer

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
