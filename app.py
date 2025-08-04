# import gradio as gr
# from qa.rag_chain import build_qa_chain, handle_greetings_and_thanks

# # Build the RAG RetrievalQA chain
# qa_chain = build_qa_chain()

# def respond(message, history):
#     greeting_thanks_response = handle_greetings_and_thanks(message)
#     if greeting_thanks_response:
#         return greeting_thanks_response

#     try:
#         response = qa_chain.invoke({"query": message})
#         answer = response.get("result", "").strip()

#         # Debugging output
#         print("Question:", message)
#         print("Raw Answer:", answer)

#         # Step 1: Extract only the part after 'Answer:' if it exists
#         if "Answer:" in answer:
#             answer = answer.split("Answer:", 1)[-1].strip()

#         # Step 2: Remove extra info if answer includes any cut-off phrases
#         cutoff_phrases = ["I don't know", "هذه اللغة غير مدعومة"]
#         for phrase in cutoff_phrases:
#             if phrase in answer:
#                 answer = answer.split(phrase)[0].strip()

#         # Step 3: Remove duplicate lines or repeated content
#         answer_lines = answer.splitlines()
#         seen = set()
#         unique_lines = []
#         for line in answer_lines:
#             line = line.strip()
#             if line and line not in seen:
#                 seen.add(line)
#                 unique_lines.append(line)
#         clean_answer = " ".join(unique_lines)

#         # Step 4: Detect language
#         def detect_lang(text):
#             arabic_chars = sum("\u0600" <= c <= "\u06FF" for c in text)
#             return "ar" if arabic_chars / max(len(text), 1) > 0.5 else "en"

#         message_lang = detect_lang(message)
#         answer_lang = detect_lang(clean_answer)

#         # Step 5: Language mismatch response
#         if message_lang != answer_lang:
#             if message_lang == "ar":
#                 return "الإجابة غير متوفرة باللغة العربية، يرجى إعادة صياغة السؤال إن أمكن."
#             else:
#                 return "The answer is not available in English. Please rephrase your question."

#         return clean_answer if clean_answer else "I don't know."

#     except Exception as e:
#         print("Error:", str(e))
#         return "Sorry, an error occurred."


# # Gradio chat interface
# chatbot_ui = gr.ChatInterface(
#     fn=respond,
#     title="📘 RAG Assistant",
#     description="Ask your question and get a concise answer based on your documents.",
#     chatbot=gr.Chatbot(),
#     textbox=gr.Textbox(placeholder="Ask your question...", container=False),
# )

# # Launch the Gradio app
# if __name__ == "__main__":
#     chatbot_ui.launch(share=True)
import json
from qa.rag_chain import build_qa_chain, handle_greetings_and_thanks

def clean_model_answer(raw_answer: str) -> str:
    # Step 1: Extract only after 'Answer:' if present
    if "Answer:" in raw_answer:
        raw_answer = raw_answer.split("Answer:", 1)[-1].strip()

    # Step 2: Remove cutoff phrases
    cutoff_phrases = ["I don't know", "هذه اللغة غير مدعومة"]
    for phrase in cutoff_phrases:
        if phrase in raw_answer:
            raw_answer = raw_answer.split(phrase)[0].strip()

    # Step 3: Remove duplicate lines
    answer_lines = raw_answer.splitlines()
    seen = set()
    unique_lines = []
    for line in answer_lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            unique_lines.append(line)
    return " ".join(unique_lines).strip()

# === Load QA dataset ===
with open("docs/iran_israel_rag_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

qa_pairs = qa_data["qa_pairs"]

qa_chain = build_qa_chain()
print("✅ Loaded QA pairs from JSON.")

results = []
for idx, qa in enumerate(qa_pairs, start=1):
    question = qa["question"]
    ground_truth = qa["answer"]

    try:
        response = qa_chain.invoke({"query": question})
        raw_answer = response.get("result", "").strip()

        # Clean to match what you see in Gradio
        model_answer = clean_model_answer(raw_answer)

        results.append({
            "id": qa["id"],
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "question_type": qa.get("question_type", ""),
            "difficulty": qa.get("difficulty", "")
        })
        print(f"[{idx}/{len(qa_pairs)}] Q: {question}")
        print(f"→ Model: {model_answer}\n")

    except Exception as e:
        print(f"Error with question {idx}: {e}")
        results.append({
            "id": qa["id"],
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": None,
            "error": str(e)
        })

# === Save results ===
output_file = "docs/qa_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"✅ Results saved to {output_file}")
