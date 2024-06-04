import os
import sys
from langchain_openai import OpenAI, OpenAIEmbeddings
# used to create the retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.chat_models import ChatOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, ServiceContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import wx
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.llms import LLMPredictor
import streamlit as st
import os
import tempfile

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

sys.path.append('../..')
#from dotenv import load_dotenv, find_dotenv = load_dotenv(find_dotenv())
OpenAI.api_key = ""
os.environ['OPENAI_API_KEY'] = OpenAI.api_key
memory_key = "history"
llm = ChatOpenAI(temperature = 0, openai_api_key=OpenAI.api_key)


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
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

def create_vector_database(methodology):
    # DB_DIR: str = os.path.join(ABS_PATH, "db")
    """
    Creates a vector database using document loaders and embeddings.
    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFace,
    and finally persists the embeddings into a Chroma vector database.
    """
    
    persist_directory = 'iam_index'
    # getting embeddings
    openAI_embeddings = OpenAIEmbeddings(openai_api_key = OpenAI.api_key)
    folder_path = r"C:\Users\EzhilPriyadharshiniK\OneDrive - Infoseck2k\Desktop\IAMtest"

    if (methodology == 'ChromaDB'):
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
    else:
        # get collection
        # initialize client
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("iam_index")

        if os.path.exists(persist_directory):
            print('i am looading existing dataaaaa')
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            sc = StorageContext.from_defaults(vector_store=vector_store)
            
            # index2 = load_index_from_storage(sc)
            # load your index from stored vectors
            vstore = VectorStoreIndex.from_vector_store(vector_store, storage_context=sc)
        else:
            print('i am creating an index')
            # documents = SimpleDirectoryReader(folder_path).load_data()
            filename_fn = lambda filename: {"file_name": filename}

            # automatically sets the metadata of each document according to filename_fn
            documents = SimpleDirectoryReader(folder_path, file_metadata=filename_fn).load_data()
            # vector_index = VectorStoreIndex.from_documents(documents)
            # vector_index.as_query_engine()
            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

            # global
            Settings.text_splitter = text_splitter
            # assign chroma as the vector_store to the context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # per-index
            vstore = VectorStoreIndex.from_documents(documents, transformations=[text_splitter], storage_context=storage_context)

            vstore.storage_context.persist("iam_index")
    return vstore

def extract_file_names(metadata_dict):
    file_names = []
    for key, value in metadata_dict.items():
        if 'file_name' in value:
            full_path = value['file_name']
    return full_path

with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

def main():   
    # if st.button('Browse'):
    #     app = wx.App()
    #     dialog = wx.DirDialog(None, "Select a folder:", style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
    #     if dialog.ShowModal() == wx.ID_OK:
    #         folder_path = dialog.GetPath() 

    st.title("📝 Upload file for Q&A")
    uploaded_file = st.file_uploader("Upload an article")
    index = create_vector_database('LLAMA_Index')
    if uploaded_file:
        directory_path = r'C:\Users\EzhilPriyadharshiniK\OneDrive - Infoseck2k\Desktop\IAMtest\test'        
        # directory_path = os.path.join(directory, uploaded_file.name)
        # with open(directory_path, "wb") as f:
        #     f.write(uploaded_file.getvalue())
        print(directory_path)

        file_metadata = lambda x : {"filename": x}
        reader = SimpleDirectoryReader(directory_path, file_metadata=file_metadata)
            
        documents = reader.load_data()
        print(type(documents))
        for d in documents:
            index.insert(document = d)

        st.write("Completed Indexing...")

    st.title("📝 Start asking question")
    query = st.chat_input("Ask a question related to IAM:")


    #if st.button("Get Answer"):
    if query:    
        
        # create a query engine
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        # Get response
        st.write(str(response))

        # Get sources
        st.write('Source File Name:')
        st.write(extract_file_names(response.metadata))

        st.write('Metadata:')
        st.write(response.metadata)

        st.write('Source Nodes:')
        st.write(response.source_nodes)
        #st.write(streaming_response.print_response_stream())
    #index.storage_context.persist(persist_dir=folder_path)
    # question = st.text_input(
    #     "Ask something about the article",
    #     placeholder="Can you give me a short summary?",
    #     disabled=not uploaded_file,
    # )


if __name__ == "__main__":
    main()



