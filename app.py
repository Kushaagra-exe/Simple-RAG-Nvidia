from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
import streamlit as st
# import os
# from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
# load_dotenv()
# api = os.getenv('NVIDIA_API')
st.sidebar.header("API Key Input")
api = st.sidebar.text_input("Enter your API key", type="password")



def embed_docs(pdf_file):
    llm = ChatNVIDIA(model='meta/llama-3.1-405b-instruct', api_key=api)
    
    import tempfile
    if "vectors" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
                temp_pdf_file.write(pdf_file.read())  # Write the uploaded file to the temp file
                temp_pdf_path = temp_pdf_file.name
        st.session_state.embeddings = NVIDIAEmbeddings(api_key=api)
        st.session_state.loaders = PyPDFLoader(temp_pdf_path)
        st.session_state.docs = st.session_state.loaders.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Provide most accurate response based on the question
    <context>{context}<context>
    Question : {input}
"""
)
st.title("Nvidia")
pdf_file = st.file_uploader("Upload a PDF document", type="pdf")

if pdf_file is not None:
    if st.button("Document Embedding"):
        if api:
            embed_docs(pdf_file)
            st.write("Vector db ready")
        else:
            st.write("Please enter your API key.")

ques_input = st.text_input("Enter the question from the document")
submit = st.button("Submit")

import time
if submit:
    if api:
        llm = ChatNVIDIA(model='meta/llama-3.1-405b-instruct', api_key=api)
        
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retreiver = st.session_state.vectors.as_retriever()
        ret_chain = create_retrieval_chain(retreiver, doc_chain)
        start = time.process_time()
        resp = ret_chain.invoke({'input': ques_input})
        
        st.write("Response time:", time.process_time()-start)
        st.write(resp['answer'])

        with st.expander("DSS"):
            for i,doc in enumerate(resp['context']):
                st.write(doc.page_content)
                st.write("------------------------------------")
    else:
        st.write("Please enter your API key.")


