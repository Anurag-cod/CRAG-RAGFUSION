from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def generate_query(question: str, generate_query_num: int):
    llm = ChatOpenAI()
    system_prompt = "You are an assistant that generates multiple search queries based on a single input query."
    human_prompt = f"""When creating queries, output each query on a new line without significantly changing the original query's meaning.
    Input query: {question}
    {generate_query_num} output queries: 
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    questions_chain = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    generated_queries = questions_chain.invoke({"question": question, "generate_query_num": generate_query_num})
    
    return generated_queries
