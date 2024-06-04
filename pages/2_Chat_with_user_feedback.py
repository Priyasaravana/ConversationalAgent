from openai import OpenAI
import streamlit as st
from streamlit_feedback import streamlit_feedback
# import trubrics
# from trubrics.integrations.streamlit import FeedbackCollector

OpenAI.api_key = ""
st.title("📝 Chat with feedback (Trubrics)")
# collector = FeedbackCollector()

"""
In this example, we're using [streamlit-feedback](https://github.com/trubrics/streamlit-feedback) and Trubrics to collect and store feedback
from the user about the LLM responses.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you? Leave feedback to help me improve!"}
    ]
if "response" not in st.session_state:
    st.session_state["response"] = None

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history) 

messages = st.session_state.messages
for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Tell me a joke about sharks"):
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    client = OpenAI(api_key=OpenAI.api_key)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    st.session_state["response"] = response.choices[0].message.content
    with st.chat_message("assistant"):
        messages.append({"role": "assistant", "content": st.session_state["response"]})
        st.write(st.session_state["response"])

if st.session_state["response"]:
    feedback_option = "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{len(messages)}",
        #key=f"feedback_{st.session_state.run_id}",
    )

    # with st.form("main", clear_on_submit=True):
    #     st.write('answer ...')
    
    #     feedback = streamlit_feedback(
    #         feedback_type="thumbs",
    #         optional_text_label="[Optional] Please provide an explanation",
    #         align="flex-start"
    #     )
    #     st.form_submit_button('save')

    st.write(f"feedback log -{feedback}")


