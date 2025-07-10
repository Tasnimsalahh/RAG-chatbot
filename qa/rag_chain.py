from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

system_prompt = "You are a helpful assistant that answers questions based on the provided context. " \
                "If the context does not contain the answer, say 'I don't know'. " \
                "If the context is not relevant, say 'This is not relevant to the question'. " \
                "Always provide a concise answer with no extra information." \
                "If the question is not clear, ask for clarification." \
                "Answer with the same language as the question, if the language is not supported,say 'this language is not supported' and answer in English."\
                
prompt_template = PromptTemplate.from_template(
        """
{system_prompt}

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""".strip()
)

CHROMA_DB_DIR = "chroma_db"

def build_qa_chain():
    # Load the vectorstore
    embedding_function = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "cuda"}
    )
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    model_name = "tiiuae/falcon-rw-1b"
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