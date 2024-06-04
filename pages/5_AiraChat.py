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
#from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
import chromadb
# used to load text
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.document_loaders.base import BaseLoader
from langchain import hub
from langchain.agents import tool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# used to create the agent executor
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.llms import CTransformers
from langchain.agents import AgentExecutor, create_openai_functions_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# used to create the prompt template
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from llama_index.core.memory import ChatMemoryBuffer
# used to create the memory
from langchain.memory import ConversationBufferMemory, DynamoDBChatMessageHistory, ConversationBufferWindowMemory

from langchain.llms import HuggingFaceHub
import torch
from transformers import BitsAndBytesConfig
import os

#other libraries
import streamlit as st
from langchain.schema import Document
import gradio as gr
import tempfile
import timeit
import datetime
import tiktoken

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
methodology = 'LI'

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
    # prompt_template = """You are an IAM assistant chatbot named "AiRA". Your expertise is 
    #     exclusively in providing information and advice about anything related to IAM. This includes flavor combinations, IAM, and general 
    #     IAM-related queries. You do not provide information outside of this 
    #     scope. If a question is not about IAM, respond with, "I specialize only in IAM related queries." .
    #     Chat History: {chat_history}
    #     Question: {question}
    #     Context: {context} 
    #     Only return the helpful answer below and nothing else.
    #     Helpful answer:
    #     """
    system_message =  """Your name is AiRA, a friendly IAM bot. 
                        You are a smart personal assistant designed to help IAM team with cyber security details.
                        Given a piece of text, you must come up with a insights of IAM.
                        Feel free to use any tools available to look up relevant information, only if neccessary                                             
                        For First answer, you should always respond with the user name and greetings 
                        Here's your conversation with the user so far:  
                        {chat_history}  
                        Now the user asked: {question}  
                        To answer this question, you need to look up from their notes about 
                         """
    

                            # When coming up with this answer, you must respond in the following format:
                        # ```
                        # {{    
                            
                        #     "AiRA answer": "$THE_ANSWER_HERE"
                        # }}
                        # ```
    # Create the prompt using create_openai_functions_agent
    memory_key = "chat_history"
    system_message_template = SystemMessagePromptTemplate.from_template(system_message)
    #prompt = ChatPromptTemplate.from_messages([system_message_template])
    
    humanTemp = """Please come up with a answer from the given documents, if not available search in the internet:"""

    prompt = ChatPromptTemplate.from_messages([
                    system_message_template,
                    #MessagesPlaceholder("chat_history", optional=True),
                    # The `variable_name` here is what must align with memory                    
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                    HumanMessagePromptTemplate.from_template("{question}"),
                    ])
    #extra_prompt_messages = [MessagesPlaceholder(variable_name=memory_key)]


    # Get the prompt to use - you can modify this!
    #prompt = hub.pull("hwchase17/openai-functions-agent")
    #prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history",  "question", "context"])
    return prompt

# @tool
# def tool(query):
#     "Searches and returns documents regarding the llm powered autonomous agents blog"
#     docs = retriever.get_relevant_documents(query)
#     return docs

def create_chain(vstore, prompt, sc):
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
    llm = ChatOpenAI(temperature = 0, openai_api_key=OpenAI.api_key)

    if methodology == 'ChromaDB':
        retriever = vstore.as_retriever()
        retriever.search_kwargs = {'k':1}
        
        qa = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True)
        #tools = [tool]
        tools = [
            Tool(
                name="doc_search_tool",
                func=qa,
                description=(
                "This tool is used to retrieve information from the knowledge base"
                )
            )
        ]
        # conversation_memory  = ConversationBufferWindowMemory(memory_key="chat_history", input_key='question', return_messages=True, output_key='output', k=5)

        #modelIAMChain = LLMChain(llm=OpenAI(), chain_type = 'stuff', retriever=retriever, prompt=prompt, memory = conversation_memory) 
        # modelIAMChain = ConversationalRetrievalChain.from_llm(
        #     llm=OpenAI(),
        #     chain_type="stuff",
        #     retriever=retriever,
        #     # return_source_documents=True,
        #     max_tokens_limit=256,
        #     # combine_docs_chain_kwargs={"prompt": prompt},
        #     #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        #     memory=conversation_memory,
        # )
    
        
        # initialize RetrievalQAModel
        # modelIAMChain = RetrievalQAWithSourcesChain.from_chain_type(llm=OpenAI(), chain_type = 'stuff', retriever=retriever, memory = conversation_memory)
    
    
        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps = True)
        #agent_executor = AgentExecutor(agent=agent, tools=tools, memory=conversation_memory, verbose=True, return_intermediate_steps = True)
        return agent_executor
    else:
        # tools = [
        #             Tool(
        #                 name="LlamaIndex",
        #                 func=lambda q: str(vstore.as_query_engine(
        #                     retriever_mode="embedding", 
        #                     verbose=True, 
        #                     service_context=sc
        #                 ).query(q)),
        #                 description="useful for when you want to answer questions about IAM. The input to this tool should be a complete english sentence.",
        #                 return_direct=True,
        #             ),
        #         ]
        

        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

        chat_engine = vstore.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=(
                "You are a chatbot named AiRA, able to have interactions about Cyber security with specialization in Identity and access Management, as well as talk. Your core expertise are Risk Management, Threat analytics and Malware reverse engineering"               
            ),
        )
        return chat_engine
   

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

def create_vector_database():  
    openAI_embeddings = OpenAIEmbeddings(openai_api_key = OpenAI.api_key)
    folder_path = r"C:\Users\EzhilPriyadharshiniK\OneDrive - Infoseck2k\Desktop\IAMtest"
    sc = ''
    if (methodology == 'ChromaDB'):
        persist_directory = 'IAM1'
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
        
        return vstore, sc
    else:
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("iam_index")
        persist_directory = 'iam_index'
    
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        sc = StorageContext.from_defaults(vector_store=vector_store)
        
        # index2 = load_index_from_storage(sc)
        # load your index from stored vectors
        vstore = VectorStoreIndex.from_vector_store(vector_store, storage_context=sc)
        return vstore, sc

def display_existing_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    else:
        for message in st.session_state["messages"]:
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

def extract_source_file_names(response):
    try:
     
        # Extract the source file names without the full path
        source_file_names = response["intermediate_steps"][0][1]["source_documents"]
        source_file_names = (doc.metadata['source'] for doc in source_file_names)
        
        # Return the list of source file names
        return source_file_names

    except KeyError as e:
        print(f"Key error: {e}")
    except TypeError as e:
        print(f"Type error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def main():   
    st.title("💬 IAM Chatbot")   
    st.write("Hello Iam Aira. Happy to Assist you. You can ask me any IAM related questions.👋")  
    st.write(" I'll do my best to assist you, but I'm still learning. If I cannot provide the information you need, don't worry! Our customer support team is monitoring my conversations. Just fill in your contact details, and we will be in touch short after. or additional help, email our super friendly human 🧠 team at support@aira.com or join our Discord community.")  
   
    # st.write(f"loaded_documents: {loaded_documents}")     
    #query = st.text_input("Ask a question:")
    
    # Replicate Credentials
    with st.sidebar:
        st.title('💬 AiRA Chatbot')    
        st.markdown('📖 Learn more about our Product [InfosecK2K](https://www.infoseck2k.com/)!')        

    display_existing_messages()

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)    
      
    prompt = set_custom_prompt()
    vstore, sc = create_vector_database()    
    agent_executor = create_chain(vstore, prompt, sc)
    query = st.chat_input("Ask a question related to IAM:")
    print(query)

    #if st.button("Get Answer"):
    if query:
        # Load model, set prompts, create vector database, and retrieve answer
        try:
            add_user_message_to_session(query)
            start = timeit.default_timer()          
            
            if methodology == 'LI':
                response = agent_executor.chat(query)
            else:
                response = agent_executor({"question": query})
                response = response['output']
            #response = modelIAMChain({"question":query})
            end = timeit.default_timer()

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
                
            st.write("Elapsed time:")
            st.write(end - start)
            # st.write(f"response: {response}") 
            # Display bot response
            # st.write("Bot Response:")
            # st.write(response)
            
        #     # Extract the source file names from the result
        #     st.write("Source File Name:")
                       
        #    # Call the function and print the results
        #     source_file_names = extract_source_file_names(response)
        #     st.write(source_file_names)            
        except Exception as e:
            st.error("No source file names found.")    

if __name__ == "__main__":
    main()