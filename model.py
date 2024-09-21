import os
import time

from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

DB_FAISS_PATH = "vectorstore/"
custom_prompt_template = """You are a MS Dhoni fan and you know everything about him. Use the context to answer the questions. Do not answer anything outside the context.

Context: {context}
Question: {question}

Provide the answer below in a clear and readable format.
Answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_type = "similarity", search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = Ollama(
        model="llama2",
        temperature=0.01,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    #db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    try:
        # Load Faiss index with dangerous deserialization enabled
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        # Use the loaded index
        # Example: query the index, etc.

    except ValueError as e:
        print(f"ValueError loading Faiss index from {DB_FAISS_PATH}: {str(e)}")
        # Handle the error appropriately (e.g., log, notify, or exit gracefully)
    except Exception as e:
        print(f"Error loading Faiss index from {DB_FAISS_PATH}: {str(e)}")

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa



#output function
def final_result(query):
    start_time = time.time()
    qa_result = qa_bot()
    response = qa_result({'query': query})
    # ANSI escape code for green color
    green_color_code = '\033[92m'

    # Reset ANSI escape code (to revert to default color)
    reset_color_code = '\033[0m'
    print("\n" + query + "\n")
    print(green_color_code + "\n" +  response['result'] + reset_color_code)
    #print(response)
    end_time = time.time()
    response_time = end_time - start_time
    print(f"Response Time:{response_time}")
    return response

final_result("when was dhoni born?")
final_result("where did Dhoni study?")
final_result("where did Dhoni's father work")
final_result("what are the awards Mahendra singh Dhoni won??")




