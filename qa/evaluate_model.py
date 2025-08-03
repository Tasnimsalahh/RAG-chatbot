import json 
from qa.rag_chain import build_qa_chain

with open("docs/iran_israel_qa.json", "r", encoding="utf-8") as f:
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
        model_answer = response.get("result", "").strip()

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

output_file = "docs/qa_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"✅ Results saved to {output_file}")