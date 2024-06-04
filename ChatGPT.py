from openai import OpenAI
import os
import sys
import streamlit as st
import replicate
from langchain.prompts import PromptTemplate

sys.path.append(r'C:\Users\EzhilPriyadharshiniK\OneDrive - Infoseck2k\Documents\Priya\GitInfosec\AiRA\RAG_LangChain')

llm = "openai/gpt-3.5-turbo:latest"
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
openai_api_key = ""
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "openai_api_key"))
client = OpenAI(api_key=openai_api_key)

with st.sidebar:
    temperature = st.sidebar.slider('temperature(Controls the randomness)', min_value=0.01, max_value=2.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p(Controls cumulative Probability distribution)', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length(Controls No. of tokens)', min_value=64, max_value=4096, value=512, step=8)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)    

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def add_user_message_to_session(prompt):
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "Hello, how can I assist you today?"  # Example dialogue string
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": string_dialogue},
    {"role": "user", "content": prompt_input},
]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, 
                                              temperature=temperature, max_tokens=max_length, top_p=top_p)  # Note: repetition_penalty is not directly supported by OpenAI API. It's a parameter used in other APIs like HuggingFace.

    return response.choices[0].message.content

if prompt := st.chat_input():    
    add_user_message_to_session(prompt)    
    st.chat_message("user").write(prompt)
    # response = client.chat.completions.create(model=llm, messages=st.session_state.messages)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):     
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
        #st.session_state.messages.append({"role": "assistant", "content": msg})
        st.session_state["messages"].append({"role": "assistant", "content": response})
        
