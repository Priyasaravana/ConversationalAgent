from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import HuggingFacePipeline
# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    GitLoader,
    NotebookLoader,
    OnlinePDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredODTLoader,
    NotebookLoader,
    UnstructuredFileLoader
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    GenerationConfig,
    TextStreamer,
    pipeline
)
from langchain.llms import HuggingFaceHub
import torch
from transformers import BitsAndBytesConfig
import os
from langchain.llms import CTransformers
import streamlit as st
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
import gradio as gr
import tempfile
import timeit

FILE_LOADER_MAPPING = {
    "csv": (CSVLoader, {"encoding": "utf-8"}),
    "doc": (UnstructuredWordDocumentLoader, {}),
    "docx": (UnstructuredWordDocumentLoader, {}),
    "epub": (UnstructuredEPubLoader, {}),
    "html": (UnstructuredHTMLLoader, {}),
    "md": (UnstructuredMarkdownLoader, {}),
    "odt": (UnstructuredODTLoader, {}),
    "pdf": (PyPDFLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "pptx": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
    "ipynb": (NotebookLoader, {}),
    "py": (PythonLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}



def load_model():
    # model_path=HuggingFaceHub(repo_id="vilsonrodrigues/falcon-7b-instruct-sharded")

    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"No model file found at {model_path}")

    # quantization_config = BitsAndBytesConfig(
    #   load_in_4bit=True,
    #   bnb_4bit_compute_dtype=torch.float16,
    #   bnb_4bit_quant_type="nf4",
    #   bnb_4bit_use_double_quant=True,
    # )

    # model_4bit = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     quantization_config=quantization_config,
    #     )

    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # pipeline = pipeline(
    #     "text-generation",
    #     model=model_4bit,
    #     tokenizer=tokenizer,
    #     use_cache=True,
    #     device_map="auto",
    #     max_length=700,
    #     do_sample=True,
    #     top_k=5,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.eos_token_id,
    # )

    # llm = HuggingFacePipeline(pipeline=pipeline)
    # llm = CTransformers(
    #     model=HuggingFaceHub(repo_id="TheBloke/Llama-2-7B-Chat-GGML", model_kwargs={"temperature":0.5, "max_length":512})
    #     # model_type=model_type,
    #     # max_new_tokens=max_new_tokens,  # type: ignore
    #     # temperature=temperature,  # type: ignore
    # )
    llm = CTransformers(
        # model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model = "TheBloke/zephyr-7B-beta-GGUF",
        # model_file = "mistral-7b-instruct-v0.1.Q8_0.gguf",
        model_file = "zephyr-7b-beta.Q4_0.gguf",
        # model="TheBloke/Llama-2-70B-chat-GGUF",
        # model = "Deci/DeciLM-6b-instruct",
        callbacks=[StreamingStdOutCallbackHandler()]
        # model_type=model_type,
        # max_new_tokens=max_new_tokens,  # type: ignore
        # temperature=temperature,  # type: ignore
    )
    return llm

# def load_document(
#     # file_path: str,
#     uploaded_files: list,
#     mapping: dict = FILE_LOADER_MAPPING,
#     default_loader: BaseLoader = UnstructuredFileLoader,
# ) -> Document:
#     loaded_documents = []
#     for uploaded_file in uploaded_files:
#         # Choose loader from mapping, load default if no match found
#         # ext = "." + uploaded_files.rsplit(".", 1)[-1]
#         ext = os.path.splitext(uploaded_file.name)[-1][1:].lower()
#         if ext in mapping:
#             loader_class, loader_args = mapping[ext]
#             loader = loader_class(uploaded_file, **loader_args)
#         else:
#             loader = default_loader(uploaded_file)
#         loaded_documents.extend(loader.load())
#     return loaded_documents

def create_vector_database(loaded_documents):
    # DB_DIR: str = os.path.join(ABS_PATH, "db")
    """
    Creates a vector database using document loaders and embeddings.
    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFace,
    and finally persists the embeddings into a Chroma vector database.
    """

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30, length_function = len)
    chunked_documents = text_splitter.split_documents(loaded_documents)

    # Initialize HuggingFace embeddings
    # embeddings = HuggingFaceEmbeddings(
    #     # model_name="sentence-transformers/all-MiniLM-L6-v2"
    #     model_name = "sentence-transformers/all-mpnet-base-v2"
    # )
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = "BAAI/bge-large-en"
    )
    
    persist_directory = 'db'
    # Create and persist a Chroma vector database from the chunked documents
    db = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=persist_directory
        # persist_directory=DB_DIR,
    )
    db.persist()
    # db = Chroma(persist_directory=persist_directory, 
    #               embedding_function=embedding)
    return db

def set_custom_prompt_condense():
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONDENSE_QUESTION_PROMPT

def set_custom_prompt():
    """
    Prompt template for retrieval for each vectorstore
    """
    # prompt_template = """<Instructions>
    # Important:
    # Answer with the facts listed in the list of sources below. If there isn't enough information below, say you don't know.
    # If asking a clarifying question to the user would help, ask the question.
    # ALWAYS return a "SOURCES" part in your answer, except for small-talk conversations.

    # Question: {question}

    # {context}


    # Question: {question}
    # Helpful Answer:

    # ---------------------------
    # ---------------------------
    # Sources:
    # """
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

# def create_chain(llm, prompt, CONDENSE_QUESTION_PROMPT, db):
def create_chain(llm, prompt, db):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.
    This function initializes a ConversationalRetrievalChain object with a specific chain type and configurations,
    and returns this  chain. The retriever is set up to return the top 3 results (k=3).
    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the 
        retriever.
    Returns:
        ConversationalRetrievalChain: The initialized conversational chain.
    """
    memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key='question', output_key='answer')
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=db.as_retriever(search_kwargs={"k": 3}),
    #     return_source_documents=True,
    #     max_tokens_limit=256,
    #     combine_docs_chain_kwargs={"prompt": prompt},
    #     condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    #     memory=memory,
    # )
    chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 3}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return chain

def create_retrieval_qa_bot(loaded_documents):
    # if not os.path.exists(persist_dir):
    #       raise FileNotFoundError(f"No directory found at {persist_dir}")

    try:
        llm = load_model()  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    try:
        prompt = set_custom_prompt()  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to get prompt: {str(e)}")

    # try:
    #     CONDENSE_QUESTION_PROMPT = set_custom_prompt_condense()  # Assuming this function exists and works as expected
    # except Exception as e:
    #     raise Exception(f"Failed to get condense prompt: {str(e)}")

    try:
        db = create_vector_database(loaded_documents)  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to get database: {str(e)}")

    try:
        # qa = create_chain(
        #     llm=llm, prompt=prompt,CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, db=db
        # )  # Assuming this function exists and works as expected
        qa = create_chain(
            llm=llm, prompt=prompt, db=db
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa

def retrieve_bot_answer(query, loaded_documents):
    """
    Retrieves the answer to a given query using a QA bot.
    This function creates an instance of a QA bot, passes the query to it,
    and returns the bot's response.
    Args:
        query (str): The question to be answered by the QA bot.
    Returns:
        dict: The QA bot's response, typically a dictionary with response details.
    """
    qa_bot_instance = create_retrieval_qa_bot(loaded_documents)
    # bot_response = qa_bot_instance({"question": query})
    bot_response = qa_bot_instance({"query": query})
    # Check if the 'answer' key exists in the bot_response dictionary
    # if 'answer' in bot_response:
    #     # answer = bot_response['answer']
    #     return bot_response
    # else:
    #     raise KeyError("Expected 'answer' key in bot_response, but it was not found.")
    # result = bot_response['answer']
    result = bot_response['result']
    sources = []
    for source in bot_response["source_documents"]:
        sources.append(source.metadata['source'])
    return result, sources

# from your_module import load_model, set_custom_prompt, set_custom_prompt_condense, create_vector_database, retrieve_bot_answer


def main():
   
    st.title("Docuverse")

    # Upload files
    uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "md", "txt", "csv", "py", "epub", "html", "ppt", "pptx", "doc", "docx", "odt", "ipynb"], accept_multiple_files=True)
    loaded_documents = []

    if uploaded_files:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as td:
            # Move the uploaded files to the temporary directory and process them
            for uploaded_file in uploaded_files:
                st.write(f"Uploaded: {uploaded_file.name}")
                ext = os.path.splitext(uploaded_file.name)[-1][1:].lower()
                st.write(f"Uploaded: {ext}")

                # Check if the extension is in FILE_LOADER_MAPPING
                if ext in FILE_LOADER_MAPPING:
                    loader_class, loader_args = FILE_LOADER_MAPPING[ext]
                    # st.write(f"loader_class: {loader_class}")

                    # Save the uploaded file to the temporary directory
                    file_path = os.path.join(td, uploaded_file.name)
                    with open(file_path, 'wb') as temp_file:
                        temp_file.write(uploaded_file.read())

                    # Use Langchain loader to process the file
                    loader = loader_class(file_path, **loader_args)
                    loaded_documents.extend(loader.load())
                else:
                    st.warning(f"Unsupported file extension: {ext}")

        # st.write(f"loaded_documents: {loaded_documents}")  
        st.write("Chat with the Document:")
        query = st.text_input("Ask a question:")

        if st.button("Get Answer"):
            if query:
                # Load model, set prompts, create vector database, and retrieve answer
                try:
                    start = timeit.default_timer()
                    llm = load_model()
                    prompt = set_custom_prompt()
                    CONDENSE_QUESTION_PROMPT = set_custom_prompt_condense()
                    db = create_vector_database(loaded_documents)
                    # st.write(f"db: {db}") 
                    result, sources = retrieve_bot_answer(query,loaded_documents)
                    end = timeit.default_timer()
                    st.write("Elapsed time:")
                    st.write(end - start)
                    # st.write(f"response: {response}") 
                    # Display bot response
                    st.write("Bot Response:")
                    st.write(result)
                    st.write(sources)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()