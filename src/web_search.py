import os
from dotenv import load_dotenv
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_core.documents import Document

load_dotenv()

os.environ["SEARCHAPI_API_KEY"]  = os.getenv("SEARCHAPI_API_KEY")

def perform_web_search(query: str):
    retriever = SearchApiAPIWrapper()
    search_results = retriever.run(query)
    
    documents = []
    if isinstance(search_results, list):
        for result in search_results:
            if isinstance(result, str):
                documents.append(Document(page_content=result))
            elif isinstance(result, Document):
                documents.append(result)
    
    return documents
