from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document

# def generate(state: dict) -> dict:
#     """
#     Generates a response based on the GraphState (state).
#     :param state: The state of the graph, which contains the LLM and the messages.
#     :return: A dictionary with the updated messages.
#     """
#     print("--- generate ---")
    
#     # Initialize LLM from state (assuming OpenAI is used here)
#     llm = ChatOpenAI()

#     # Retrieve the messages from the state
#     messages = state["messages"]
    
#     # Invoke the LLM with the messages to generate the response
#     response = llm.invoke(messages)
    
#     print("--- end ---\n")
    
#     # Return the response wrapped in a dictionary, as in the original logic
#     return {"messages": [response]}

def generate(state: dict) -> dict:
    """
    Generates a response based on the GraphState (state).
    :param state: The state of the graph, which contains the LLM and the messages.
    :return: A dictionary with the updated messages.
    """
    print("--- generate ---")
    
    # Initialize LLM from state (assuming OpenAI's Chat API is used here)
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # Specify the model you are using
    
    # Retrieve the messages from the state
    messages = state["messages"]
    
    # Prepare messages for the LLM (convert Document objects to plain text if necessary)
    processed_messages = []
    for message in messages:
        if isinstance(message, Document):  # If it's a Document, extract the text
            processed_messages.append(message.page_content)
        else:
            processed_messages.append(str(message))  # If it's a string, just append it
    
    # Create a single string or multiple inputs depending on how the LLM expects it
    prompt = "\n\n".join(processed_messages)  # Join the content with newlines (or adjust as needed)
    
    # Invoke the LLM with the processed messages
    response = llm.invoke([prompt])  # Pass the prompt as a list to match the expected format
    
    print("--- end ---\n")
    
    # Return the response wrapped in a dictionary, as in the original logic
    return {"messages": [response]}
