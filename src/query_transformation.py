from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

def transform_query(query: str):
    llm = ChatOpenAI()
    system_prompt = "You are an assistant that refines user queries based on certain rules."
    human_prompt = f"Input query: {query}\nTransformed query: "
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    transformation_chain = prompt | llm
    transformed_query = transformation_chain.invoke({"query": query})
    
    return transformed_query
