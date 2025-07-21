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
        print("Raw Answer:", answer)

        # Remove anything after the first "I don't know" if it exists
        cutoff_phrases = ["I don't know", "هذه اللغة غير مدعومة"]
        for phrase in cutoff_phrases:
            if phrase in answer:
                answer = answer.split(phrase)[0].strip()

        # Remove duplicated or repeated sections if any
        answer_lines = answer.splitlines()
        unique_lines = []
        for line in answer_lines:
            if line.strip() and line.strip() not in unique_lines:
                unique_lines.append(line.strip())
        clean_answer = " ".join(unique_lines)

        # Language check
        def detect_lang(text):
            arabic_chars = sum("\u0600" <= c <= "\u06FF" for c in text)
            return "ar" if arabic_chars / max(len(text), 1) > 0.5 else "en"

        message_lang = detect_lang(message)
        answer_lang = detect_lang(clean_answer)

        if message_lang != answer_lang:
            if message_lang == "ar":
                return "الإجابة غير متوفرة باللغة العربية، يرجى إعادة صياغة السؤال إن أمكن."
            else:
                return "The answer is not available in English. Please rephrase your question."

        return clean_answer

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
