import streamlit as st

# Define a health check endpoint
def health_check():
    st.write("status: healthy")

# Define your main application function
def main():
    st.title("Your Streamlit Application")
    st.write("Your application content goes here")

# Streamlit's routing based on the query parameters
if st.experimental_get_query_params().get("health", [""])[0] == "check":
    health_check()
else:
    main()
