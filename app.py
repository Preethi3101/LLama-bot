import streamlit as st
import replicate
import os
from PyPDF2 import PdfFileReader
from typing import List

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfFileReader(uploaded_file)
    text = ''
    for page_num in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page_num).extractText()
    return text

# Upload multiple PDF files
uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_texts = []
    for file in uploaded_files:
        pdf_texts.append(extract_text_from_pdf(file))
    st.success("PDF files uploaded successfully.")

    # Store PDF texts in session state
    st.session_state.pdf_texts = pdf_texts

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":0.7, "top_p":0.9, "max_length":120, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.text_input("Ask a question about the uploaded PDF(s)", key="pdf_question", disabled=not replicate_api):
    if "pdf_texts" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response based on PDF texts and user prompt
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                combined_text = " ".join(st.session_state.pdf_texts)
                response = generate_llama2_response(combined_text + " " + prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
    else:
        st.warning("Please upload PDF files first.")

# Clear chat history
if st.button('Clear Chat History'):
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
