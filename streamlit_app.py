# import os
# import streamlit as st
# from dotenv import load_dotenv
# from src.pdf_processing import process_pdf
# from src.query_generation import generate_query
# from src.web_search import perform_web_search
# from src.llm_integration import integrate_question, create_llm_response, grade_documents
# from src.document_fusion import fuse_documents
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS


# # Load environment variables
# load_dotenv()

# # Load API keys from environment variables
# openai_api_key = os.getenv("OPENAI_API_KEY")
# search_api_key = os.getenv("SEARCHAPI_API_KEY")

# # Ensure the keys are set correctly
# if not openai_api_key:
#     st.error("OPENAI_API_KEY environment variable not set.")
# if not search_api_key:
#     st.error("SEARCHAPI_API_KEY environment variable not set.")

# # Set API keys for LangChain and OpenAI
# os.environ["OPENAI_API_KEY"] = openai_api_key if openai_api_key else "default_openai_key"
# os.environ["SEARCHAPI_API_KEY"] = search_api_key if search_api_key else "default_searchapi_key"

# # Streamlit app layout
# st.title("CorrectiveRAG+ RAGFusion: AI Document Search")

# # Upload PDF
# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
# if uploaded_file is not None:
#     # Save PDF to disk
#     with open("uploaded_file.pdf", "wb") as f:
#         f.write(uploaded_file.read())
    
#     # Load and process the PDF
#     documents = process_pdf("uploaded_file.pdf")
#     st.write(f"Loaded {len(documents)} chunks from the PDF.")

# # Input query from user
# query = st.text_input("Enter your query")

# # Define the decide_to_generate function
# def decide_to_generate(state):
#     st.write("--- decide_to_generate ---")
#     is_search = state.get('is_search', False)  # Safely access 'is_search' key
#     if is_search:
#         return "transform_query"
#     else:
#         return "create_message"

# # Button to generate queries and search
# if st.button("Search and Generate"):
#     if uploaded_file is not None and query:
#         st.write("Generating queries...")
        
#         # Generate search queries
#         generated_queries = generate_query(query, 4)
#         st.write("### Generated Queries")
#         st.write("\n".join(f"- {q}" for q in generated_queries))
        
#         # Integrate queries
#         integrated_query = integrate_question(generated_queries)
#         st.write("### Integrated Query")
#         st.write(integrated_query)
        
#         # Perform similarity search using FAISS
#         embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#         vectordb = FAISS.from_documents(documents, embeddings)
        
#         fusion_documents = []
#         for question in generated_queries:
#             docs = vectordb.similarity_search(question, k=3)
#             fusion_documents.append(docs)
        
#         # Fuse documents
#         fused_docs = fuse_documents(fusion_documents)
        
#         # Grade documents based on relevance
#         graded_docs = grade_documents(integrated_query, fused_docs)
        
#         # # Print filtered documents in the app
#         # st.write("### Graded documents:")
#         # for doc in graded_docs["documents"]:
#         #     st.write(f"Graded documents: {doc.page_content[:500]}...")  # Display first 500 characters

#         # Check if further web search is needed
#         if graded_docs["is_search"]:
#             st.write("Performing web search...")
#             web_results = perform_web_search(integrated_query)
#             graded_docs["documents"].extend(web_results)
#             st.write("### Web Search Results:")
#             for doc in web_results:
#                 st.write(f"Web Result Content: {doc.page_content[:500]}...")  # Display first 500 characters

#             st.write("### Graded Document + Web search")
#             for doc in graded_docs["documents"]:
#                 st.write(f"Document Content: {doc.page_content[:500]}...")  # Display first 500 characters
        
#         # Generate final response
#         response = create_llm_response(query, graded_docs["documents"])
#         response_content = response.content
        
#         st.write("### Generated Response")
#         st.markdown(response_content)  # Use markdown to render properly formatted response including LaTeX equations
#     else:
#         st.write("Please upload a PDF and enter a query.")
import os
import streamlit as st
from dotenv import load_dotenv
from src.pdf_processing import process_pdf
from src.query_generation import generate_query
from src.web_search import perform_web_search
from src.llm_integration import integrate_question, grade_documents
from src.document_fusion import fuse_documents
from src.query_transformation import transform_query  # New Import
from src.response_generation import generate  # New Import
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document


# Load environment variables
load_dotenv()

# Load API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
search_api_key = os.getenv("SEARCHAPI_API_KEY")

# Ensure the keys are set correctly
if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable not set.")
if not search_api_key:
    st.error("SEARCHAPI_API_KEY environment variable not set.")

# Set API keys for LangChain and OpenAI
os.environ["OPENAI_API_KEY"] = openai_api_key if openai_api_key else "default_openai_key"
os.environ["SEARCHAPI_API_KEY"] = search_api_key if search_api_key else "default_searchapi_key"

# Streamlit app layout
st.title("CorrectiveRAG + RAGFusion: AI Document Search")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Save PDF to disk
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and process the PDF
    documents = process_pdf("uploaded_file.pdf")
    st.write(f"Loaded {len(documents)} chunks from the PDF.")

# Input query from user
query = st.text_input("Enter your query")

# Define the decide_to_generate function
def decide_to_generate(state):
    st.write("--- decide_to_generate ---")
    is_search = state.get('is_search', False)  # Safely access 'is_search' key
    if is_search:
        return "transform_query"
    else:
        return "create_message"

# Inside your search button logic
if st.button("Search and Generate"):
    if uploaded_file is not None and query:
        # st.write("Transforming the query...")

        # # Step 1: Transform the query (optional)
        # transformed_query = transform_query(query)
        # st.write("### Transformed Query")
        # st.write(transformed_query)

        # Step 2: Generate search queries
        st.write("Generating queries...")
        generated_queries = generate_query(query, 4)
        st.write("### Generated Queries")
        st.write("\n".join(f"- {q}" for q in generated_queries))

        # Step 3: Integrate queries
        integrated_query = integrate_question(generated_queries)
        st.write("### Integrated Query")
        st.write(integrated_query)

        # Step 4: Perform similarity search using FAISS
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectordb = FAISS.from_documents(documents, embeddings)

        fusion_documents = []
        for question in generated_queries:
            docs = vectordb.similarity_search(question, k=3)
            fusion_documents.append(docs)

        # Step 5: Fuse documents
        fused_docs = fuse_documents(fusion_documents)

        # Step 6: Grade documents based on relevance
        graded_docs = grade_documents(integrated_query, fused_docs)
        st.write("### Graded Documents")
        for doc in graded_docs["documents"]:
            st.write(f"Document Content: {doc.page_content[:500]}...")

        # Step 7: Decide whether to transform the query or create a message
        next_step = decide_to_generate(graded_docs)

        if next_step == "transform_query":
            # Transform the query for web search
            st.write("Transforming query for web search...")
            transformed_query = transform_query(integrated_query)
            st.write("### Transformed Query for Web Search")
            st.write(transformed_query)

            # Step 8: Perform web search
            st.write("Performing web search...")
            web_results = perform_web_search(transformed_query)
            graded_docs["documents"].extend(web_results)
            st.write("### Web Search Results:")
            for doc in web_results:
                st.write(f"Web Result Content: {doc.page_content[:500]}...")

        # Step 9: Create final message
        st.write("Creating the final message...")
        state = {
            "messages": graded_docs["documents"]
        }
        response_data = generate(state)
        
        # Step 10: Display the generated response
        response = response_data["messages"][0]
        st.write("### Generated Response")
        st.markdown(response.content)
    else:
        st.write("Please upload a PDF and enter a query.")


# import os
# import streamlit as st
# from dotenv import load_dotenv
# from src.pdf_processing import process_pdf
# from src.query_generation import generate_query
# from src.web_search import perform_web_search
# from src.llm_integration import integrate_question, grade_documents
# from src.document_fusion import fuse_documents
# from src.query_transformation import transform_query
# from src.response_generation import generate
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# # from langgraph.graph import StateGraph, END
# from langchain.schema import Document
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, END
# from typing import TypedDict

# # Load environment variables
# load_dotenv()

# # Load API keys from environment variables
# openai_api_key = os.getenv("OPENAI_API_KEY")
# search_api_key = os.getenv("SEARCHAPI_API_KEY")

# # Ensure the keys are set correctly
# if not openai_api_key:
#     st.error("OPENAI_API_KEY environment variable not set.")
# if not search_api_key:
#     st.error("SEARCHAPI_API_KEY environment variable not set.")

# # Set API keys for LangChain and OpenAI
# os.environ["OPENAI_API_KEY"] = openai_api_key if openai_api_key else "default_openai_key"
# os.environ["SEARCHAPI_API_KEY"] = search_api_key if search_api_key else "default_searchapi_key"

# # Streamlit app layout
# st.title("CorrectiveRAG + RAGFusion: AI Document Search")

# # Initialize the state graph
# graph = StateGraph({
#     'llm_openai': ChatOpenAI(),
#     'emb_model': OpenAIEmbeddings(model="text-embedding-3-large"),
#     'question': '',
#     'generate_querys': [],
#     'generate_query_num': 4,
#     'integration_question': '',
#     'transform_question': '',
#     'messages': [],
#     'fusion_documents': [],
#     'documents': [],
#     'is_search': False
# })
# graph.set_entry_point("generate_query")
# graph.add_node("generate_query", generate_query)
# graph.add_edge("generate_query", "retrieve")
# graph.add_node("retrieve", retrieve)
# graph.add_edge("retrieve", "fusion")
# graph.add_node("fusion", fusion)
# graph.add_edge("fusion", "integration_query")
# graph.add_node("integration_query", integration_query)
# graph.add_edge("integration_query", "grade_documents")
# graph.add_node("grade_documents", grade_documents)
# graph.add_conditional_edges(
#     "grade_documents",
#     decide_to_generate,
#     {
#         "transform_query": "transform_query",
#         "create_message": "create_message"
#     }
# )
# graph.add_node("transform_query", transform_query)
# graph.add_edge("transform_query", "web_search")
# graph.add_node("web_search", web_search)
# graph.add_edge("web_search", "create_message")
# graph.add_node("create_message", create_message)
# graph.add_edge("create_message", "generate")
# graph.add_node("generate", generate)
# graph.add_edge("generate", END)
# compile_graph = graph.compile()

# # Upload PDF
# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
# if uploaded_file is not None:
#     # Save PDF to disk
#     with open("uploaded_file.pdf", "wb") as f:
#         f.write(uploaded_file.read())
    
#     # Load and process the PDF
#     documents = process_pdf("uploaded_file.pdf")
#     st.write(f"Loaded {len(documents)} chunks from the PDF.")

# # Input query from user
# query = st.text_input("Enter your query")

# # Search and generate button logic
# if st.button("Search and Generate"):
#     if uploaded_file is not None and query:
#         st.write("Processing the query...")

#         # Initialize state
#         state = {
#             'llm_openai': ChatOpenAI(),
#             'emb_model': OpenAIEmbeddings(model="text-embedding-3-large"),
#             'question': query,
#             'generate_querys': [],
#             'generate_query_num': 4,
#             'integration_question': '',
#             'transform_question': '',
#             'messages': [],
#             'fusion_documents': [],
#             'documents': documents,
#             'is_search': False
#         }

#         # Stream through the graph and print the result at each step
#         final_output = None
#         for output in compile_graph.stream(state):
#             for key, value in output.items():
#                 st.write(f"Node '{key}':")
#                 st.write(value)
                
#                 # Capture the final output when the node 'generate' is reached
#                 if key == "generate":
#                     final_output = value["messages"]

#         # Display the final response
#         if final_output:
#             st.write("### Generated Response")
#             st.markdown(final_output[0].content)
#     else:
#         st.write("Please upload a PDF and enter a query.")
