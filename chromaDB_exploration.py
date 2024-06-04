'''
References
https://medium.com/@varsha.rainer/document-loaders-in-langchain-7c2db9851123

'''
import os
#from openai import OpenAI 
import openai
import sys
import pdf2image
from pdf2image import convert_from_path
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
#from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI, OpenAIEmbeddings
import tiktoken
from deta import Deta
from weasyprint import HTML
import weasyprint
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
import datetime
from urllib.parse import urlparse

sys.path.append('../..')
#from dotenv import load_dotenv, find_dotenv = load_dotenv(find_dotenv())
OpenAI.api_key = ""
os.environ['OPENAI_API_KEY'] = OpenAI.api_key


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

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

def read_files_from_folder(folder_path):
    file_data = []
    loaded_documents = []
    for file_name in os.listdir(folder_path):
        print(file_name)
        ext = os.path.splitext(file_name)[-1][1:].lower()   
        print(ext)
        if ext in FILE_LOADER_MAPPING:
            loader_class, loader_args = FILE_LOADER_MAPPING[ext]

            # Save the uploaded file to the temporary directory
            file_path = os.path.join(folder_path, file_name)
                        
            # Use Langchain loader to process the file
            loader = loader_class(file_path, **loader_args)
            loaded_documents.extend(loader.load())
        else:
            print(f"Unsupported file extension: {ext}")
    return loaded_documents

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

def create_vector_database():
    # DB_DIR: str = os.path.join(ABS_PATH, "db")
    """
    Creates a vector database using document loaders and embeddings.
    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFace,
    and finally persists the embeddings into a Chroma vector database.
    """
    
    persist_directory = 'dbIAM1'
    # getting embeddings
    openAI_embeddings = OpenAIEmbeddings(openai_api_key = OpenAI.api_key)
    folder_path = r"C:\Users\EzhilPriyadharshiniK\OneDrive - Infoseck2k\Desktop\IAMtest"

    if os.path.exists(persist_directory):
        #shutil.rmtree(persist_directory)        
        vstore = Chroma(persist_directory=persist_directory, embedding_function=openAI_embeddings)       
    else:  
        loaded_documents = read_files_from_folder(folder_path)  
         # chunking
        char_text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        doc_texts = char_text_splitter.split_documents(loaded_documents)
        # set up chroma DB
        vstore = Chroma.from_documents(doc_texts, embedding=openAI_embeddings, persist_directory=persist_directory)
        vstore.persist()
    return vstore

def create_chain(vstore, prompt):
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
    # memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key='question', output_key='answer')
    # # chain = ConversationalRetrievalChain.from_llm(
    # #     llm=llm,
    # #     chain_type="stuff",
    # #     retriever=db.as_retriever(search_kwargs={"k": 3}),
    # #     return_source_documents=True,
    # #     max_tokens_limit=256,
    # #     combine_docs_chain_kwargs={"prompt": prompt},
    # #     condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    # #     memory=memory,
    # # )
    # chain = RetrievalQA.from_chain_type(llm=llm,
    #                                    chain_type='stuff',
    #                                    retriever=db.as_retriever(search_kwargs={'k': 3}),
    #                                    return_source_documents=True,
    #                                    chain_type_kwargs={'prompt': prompt}
    #                                    )
    retriever = vstore.as_retriever()
    retriever.search_kwargs = {'k':1}

    # initialize RetrievalQAModel
    modelIAMChain = RetrievalQAWithSourcesChain.from_chain_type(llm=OpenAI(), chain_type = 'stuff', retriever=retriever)
    return modelIAMChain

def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    else:
        for message in st.session_state["messages"]:
            print('display message')
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def add_user_message_to_session(prompt):
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

def add_to_database(query, response):
    # db = deta.Base("topical_q_a")
    # timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    # db.put({"query": query, "response": response, "timestamp": timestamp})
    return query, response


def print_markdown_from_file(file_path):
    with open(file_path, "r") as f:
        markdown_content = f.read()
        st.markdown(markdown_content)

def create_pdfs(url, output_directory):
    domain = urlparse(url).netloc
    domain = domain.replace(".", "")
    pdf = weasyprint.HTML(url).write_pdf()
    open(os.path.join(output_directory, domain+'.pdf'), 'wb').write(pdf)

prompt = set_custom_prompt()
vstore = create_vector_database()

modelIAMChain = create_chain(vstore, prompt)

def main():   
    st.title("💬 IAM Chatbot")   
    st.write("Hello Iam Aira. Happy to Assist you. You can ask me any IAM related questions.👋")    
   
    # st.write(f"loaded_documents: {loaded_documents}")     
    #query = st.text_input("Ask a question:")
    
    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Llama 2 Chatbot')    
        st.markdown('📖 Learn more about our Product [InfosecK2K](https://www.infoseck2k.com/)!')

    display_existing_messages()

    query = st.chat_input("Ask a question related to IAM:")
    print(query)
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    directory = r'C:\Users\EzhilPriyadharshiniK\OneDrive - Infoseck2k\Desktop\IAMtest'
    url = "https://en.wikipedia.org/wiki/Customer_identity_access_management"

    if not os.path.exists(directory):
        os.makedirs(directory)
    create_pdfs(url, directory)
    
    #if st.button("Get Answer"):
    if query:
        # Load model, set prompts, create vector database, and retrieve answer
        try:
            add_user_message_to_session(query)
            start = timeit.default_timer()          
            
            response = modelIAMChain({"question":query})
            end = timeit.default_timer()

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown(response['answer'])
                st.session_state["messages"].append({"role": "assistant", "content": response['answer']})
                
            st.write("Elapsed time:")
            st.write(end - start)
            # st.write(f"response: {response}") 
            # Display bot response
            st.write("Bot Response:")
            st.write(response['answer'])
            st.write(response['sources'])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")    

if __name__ == "__main__":
    main()