import os
import sys
from langchain_openai import OpenAI, OpenAIEmbeddings
# used to create the retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import wx
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

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

class DocLoader:
    @staticmethod
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
            if os.path.exists(persist_directory):
                # initialize client
                db = chromadb.PersistentClient(path="./chroma_db")

                # get collection
                chroma_collection = db.get_or_create_collection("iam_index")

                # assign chroma as the vector_store to the context
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # load your index from stored vectors
                vstore = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            else:
                # documents = SimpleDirectoryReader(folder_path).load_data()
                filename_fn = lambda filename: {"file_name": filename}

                # automatically sets the metadata of each document according to filename_fn
                documents = SimpleDirectoryReader(folder_path, file_metadata=filename_fn).load_data()
                # vector_index = VectorStoreIndex.from_documents(documents)
                # vector_index.as_query_engine()
                text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

                # global
                Settings.text_splitter = text_splitter

                # per-index
                index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])

                index.storage_context.persist("iam_index")
        return vstore

