import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

st.title("🦜🔗 AiRA - Recommendation Engine")

OpenAI.api_key = ""
os.environ['OPENAI_API_KEY'] = OpenAI.api_key


def blog_outline(topic):
    # Instantiate LLM model
    llm = OpenAI()
    # Prompt
    template = "As an experienced AI engine named AiRA, generate an recommendation for a cyber security company to perform risk assessment about {topic}."
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=topic)
    # Run LLM model
    response = llm(prompt_query)
    # Print results
    return st.info(response)


with st.form("myform"):
    topic_text = st.text_input("Enter prompt:", "")
    submitted = st.form_submit_button("Submit")
    blog_outline(topic_text)
