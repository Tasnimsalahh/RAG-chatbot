from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

system_prompt = "You are a helpful assistant that answers questions based on the provided context and the language of the query. " \
                "If the context does not contain the answer, say 'I don't know'. " \
                "If the context is not relevant, say 'This is not relevant to the question'. " \
                "Always provide a concise answer with no extra information." \
                "If the question is not clear, ask for clarification." \

prompt_template = PromptTemplate.from_template(
        """
{system_prompt}

You must use only the following context to answer the question. keep your answers straight to the point and concise. Don't add any extra information or explanations.\
    Answer with the same language as the question, if the language is not supported,say 'this language is not supported' and answer in English.\
    don't use any other language than the one used in the question.\
    اذا جاء السؤال باللغة العربية, اجب باللغة العربية. اذا جاء السؤال باللغة الانجليزية, اجب باللغة الانجليزية.\
    اذا جاء السؤال بلغة غير العربية او الانجليزية, قل 'هذه اللغة غير مدعومة' و اجب باللغة الانجليزية.

Context:
{context}

Question:
{question}

Answer:
""".strip()
)
def handle_greetings_and_thanks(question: str) -> str | None:
        greetings_arabic = [
            "مرحبا", "مرحباً", "أهلاً","اهلاً","اهلا","أهلا", "السلام عليكم", "السّلام عليكم", "أهلاً وسهلاً", "أهلا وسهلا",
            "صباح الخير", "صَباحُ الخَيْر", "مساء الخير", "مَساءُ الخَيْر", "تحية طيبة", "تحيّة طيّبة", 
            "حيّاك الله", "حيّاكم الله", "سلام عليكم", "سلامٌ عليكم"
        ]

        thanks_arabic = [
            "شكرا", "شكرًا", "أشكرك", "أَشكُرُك","شكرا", "جزاك الله خيرا", "جزاكَ اللهُ خيرًا", "جزاكي الله خيرا",
            "جزاكم الله خيرا", "ممتن", "مُمتن", "ممتنة", "مُمتنّة", "شكرًا جزيلاً", "شكرا جزيلا", 
            "كل الشكر", "ألف شكر", "شكرًا لك", "بارك الله فيك", "بارك الله فيكم"
        ]
        greetings_english = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        thanks_english = ["thank you", "thanks", "appreciate it", "much appreciated", "grateful", "thank you very much"]

        stripped_question = question.strip().lower()
        if any(greet in stripped_question for greet in greetings_arabic):
            return "مرحبًا بك! كيف يمكنني مساعدتك اليوم؟"
        elif any(thank in stripped_question for thank in thanks_arabic):
            return "على الرحب والسعة! لا تتردد في طرح أي سؤال آخر."
        elif any(greet in stripped_question for greet in greetings_english):
            return "Hello! How can I assist you today?"
        elif any(thank in stripped_question for thank in thanks_english):
            return "You're welcome! Feel free to ask any other questions."
        return None


CHROMA_DB_DIR = "chroma_db"

def build_qa_chain():
    # Load the vectorstore
    embedding_function = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "cuda"}
    )
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True
    )

    pipeline_instance = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2
    )
    llm = HuggingFacePipeline(pipeline=pipeline_instance)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template.partial(system_prompt=system_prompt)
        }
    )

    return qa_chain