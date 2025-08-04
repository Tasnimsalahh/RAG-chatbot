# import json 
# from qa.rag_chain import build_qa_chain

# with open("docs/iran_israel_rag_qa.json", "r", encoding="utf-8") as f:
#     qa_data = json.load(f)

# qa_pairs = qa_data["qa_pairs"]

# qa_chain = build_qa_chain()
# print("✅ Loaded QA pairs from JSON.")

# results = []
# for idx, qa in enumerate(qa_pairs, start=1):
#     question = qa["question"]
#     ground_truth = qa["answer"]
    
#     try:
#         response = qa_chain.invoke({"query": question})
#         model_answer = response.get("result", "").strip()

#         results.append({
#             "id": qa["id"],
#             "question": question,
#             "ground_truth": ground_truth,
#             "model_answer": model_answer,
#             "question_type": qa.get("question_type", ""),
#             "difficulty": qa.get("difficulty", "")
#         })
#         print(f"[{idx}/{len(qa_pairs)}] Q: {question}")
#         print(f"→ Model: {model_answer}\n")
#     except Exception as e:
#         print(f"Error with question {idx}: {e}")
#         results.append({
#             "id": qa["id"],
#             "question": question,
#             "ground_truth": ground_truth,
#             "model_answer": None,
#             "error": str(e)
#         })

# output_file = "docs/qa_results.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)

# print(f"✅ Results saved to {output_file}")

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
