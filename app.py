import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llama import LlamaEmbeddings  # Import LlamaEmbeddings
import streamlit as st
import replicate
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
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

# Function to split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Function to generate LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a professional in understanding content from PDFs and answering any questions related to it. Understand the content provided and answer the questions appropriately. Ensure that the sentences you provide are grammatically correct. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, 'answer is not available in the context', don't provide the wrong answer.\n\n"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += f"User: {dict_message['content']}\n\n"
        else:
            string_dialogue += f"Assistant: {dict_message['content']}\n\n"

    output = replicate.run('allenai/longformer-qa-large-finetuned-squad', 
                           input={"prompt": f"{string_dialogue} {prompt_input}"},
                           temperature=0.7, top_p=0.9, max_length=120, repetition_penalty=1)
    return output

# User-provided prompt
if prompt := st.text_input("Ask a question about the uploaded PDF(s)", key="pdf_question"):
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
