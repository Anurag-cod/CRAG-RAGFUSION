import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

def process_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    return documents
