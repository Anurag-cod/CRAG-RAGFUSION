from typing import List
import streamlit as st
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser


def integrate_question(generated_queries: list):
    llm = ChatOpenAI()
    system_prompt = "You are a question rewriter that consolidates multiple input questions into one question."
    human_prompt = f"Please output only the integrated question.\nMultiple questions: {generated_queries}\nIntegrated question:"
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    integration_chain = prompt | llm | StrOutputParser()
    
    integrated_query = integration_chain.invoke({"query": "\n".join(generated_queries)})
    return integrated_query


def create_llm_response(query: str, documents: List[Document]) -> str:
    print("--- create_llm_response ---")
    
    # Initialize the LLM (e.g., OpenAI's ChatGPT)
    llm = ChatOpenAI()
    
    # Prepare the prompt template
    # system_message = "You will always respond in English."
    system_message="You will look at context, query and answer from both."
    human_message = """Refer to the context separated by '=' signs below to answer the question.

    {context}

    Question: {query}
    """
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_message),
        ]
    )
    
    # Prepare the context and question
    partition = "\n" + "=" * 20 + "\n"
    documents_context = partition.join([doc.page_content for doc in documents if hasattr(doc, 'page_content')])
    
    # Format the messages
    messages = prompt.format_messages(context=documents_context, query=query)
    
    # Invoke the prompt
    response = llm.invoke(messages)
    
    # Return the response
    return response


def grade_documents(query, documents):
    llm = ChatOpenAI()
    system_prompt = """
    You are an assistant that evaluates the relevance between searched documents and user questions.
    If the document contains keywords or semantic content related to the question, you evaluate it as relevant.
    Respond with "Yes" for relevance and "No" for no relevance.
    """
    
    human_prompt = """
    Document: {context}
    Question: {query}
    Relevance ("Yes" or "No"): """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    
    filtered_docs = []
    is_search = False
    grade_chain = prompt | llm | StrOutputParser()
    
    st.write("### Relavance Evaluation by Integrated Query")
    
    for doc in documents:
        grade = grade_chain.invoke({"context": doc.page_content, "query": query})
        st.write(f"Document: {doc.page_content[:500]}...")
        st.markdown(f"**Relevance:** {grade}")
        if "Yes" in grade:
            filtered_docs.append(doc)
            
        else:
            is_search = True
    
    return {"documents": filtered_docs, "is_search": is_search}
