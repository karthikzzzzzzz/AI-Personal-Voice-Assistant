from langchain_community.document_loaders import PyPDFLoader 
import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import asyncio

load_dotenv()

openai_key=st.secrets["openai_key"]

chroma_client = chromadb.PersistentClient(path="./knowledge_base")
collection_name="FAQs"
collection = chroma_client.get_or_create_collection(collection_name)

async def load_multiple_pdfs(file_paths):
    all_pages = []
    for file_path in file_paths:
        try:
            loader = PyPDFLoader(file_path)
            pages = await loader.aload()  
            all_pages.extend(pages)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return all_pages

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200,  
    length_function=len,
)


def generate_store_embeddings(chunks):
    embedding_model = OpenAIEmbeddings(api_key=openai_key)
    embeddings = embedding_model.embed_documents(chunks)
    for chunk_id, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": "knowledge_base"}],
            ids=[str(chunk_id)] 
        )

async def initialize_knowledge_base():
    pdf_files = ['./short.pdf','./story.pdf','./project-3.pdf','./resume.pdf']
    pages = await load_multiple_pdfs(pdf_files) 

    chunks = []
    for page in pages:
        chunks.extend(text_splitter.split_text(page.page_content))
        
    generate_store_embeddings(chunks)   

if __name__=="__main__":
    asyncio.run(initialize_knowledge_base())



   


  




        

