from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate , ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based only on the provided context and in the same language as the question. "
               "If the context does not contain the answer, say: 'I don't know'. "
               "If the question is not relevant to the context, say: 'This is not relevant to the question'. "
               "Answer concisely. Do not provide explanations or extra information. "
               "Only use the language of the question (Arabic or English). "
               "If the language is not supported, say: 'هذه اللغة غير مدعومة' and continue in English."),
    ("human", """You must use only the following context to answer the question. 
Answer with the same language as the question.
Don't add any extra information.

Context:
{context}

Question:
{question}

Answer:""")
])

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
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype="auto",
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True
    )

    pipeline_instance = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.3,
        repetition_penalty=1.2
    )
    llm = HuggingFacePipeline(pipeline=pipeline_instance)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template
        }
    )

    return qa_chain