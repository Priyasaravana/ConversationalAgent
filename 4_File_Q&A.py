import streamlit as st
from langchain_openai import OpenAI
import os
from ..VectorIndexing import DocLoader
# from ..utils import VectorIndexing as vecIndexing
# from ..utils import DocLoader

OpenAI.api_key = ""
os.environ['OPENAI_API_KEY'] = OpenAI.api_key

with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"


# if st.button('Browse'):
#     app = wx.App()
#     dialog = wx.DirDialog(None, "Select a folder:", style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
#     if dialog.ShowModal() == wx.ID_OK:
#         folder_path = dialog.GetPath() 

st.title("📝 Upload file for Q&A")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
st.write('start the querying engine')
index = VectorIndexing.create_vector_database('LLAMA_Index')
# create a query engine
query_engine = index.as_query_engine()
response = query_engine.query("What is llama2?")
st.write(response)

#index.storage_context.persist(persist_dir=folder_path)
# question = st.text_input(
#     "Ask something about the article",
#     placeholder="Can you give me a short summary?",
#     disabled=not uploaded_file,
# )

st.write("Folder location indexed and ready for querying")



