import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Set your OpenAI API key here
os.environ['OPENAI_API_KEY'] = "" # À remplacer par votre clef d'API OpenAI disponible sur https://platform.openai.com/api-keys

def process_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    return texts

def setup_embeddings_and_vector_db(texts, persist_directory="./storage"):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    
    return vectordb

def main():
    st.title("Retrieval Augmented Generation (RAG) :")
    st.title("PDF Content-based Q&A with LangChain and GPT-4")
    
    pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])
    question = st.text_input("Enter your question:")
    
    if pdf_file is not None and question:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(pdf_file.getvalue())
        
        texts = process_pdf("temp_uploaded.pdf")
        vectordb = setup_embeddings_and_vector_db(texts)
        
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model_name='gpt-4')
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        query = f"###Prompt {question}"
        llm_response = qa(query)
        answer = llm_response["result"]
        
        # Display the answer
        st.write("Answer:", answer)

if __name__ == '__main__':
    main()