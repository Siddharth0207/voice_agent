from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware  # Import for handling CORS
from fastapi import APIRouter
# Initialize FastAPI app
router = APIRouter()


# This will be used for chat purpose if the user wants to chat instead of Voice  
@router.get("/api/summary/{topic}")
async def get_summary(topic: str):  # function name can be changed to something more descriptive later on based on purpose
    """
    Generate two summaries about a given topic using an LLM chain.

    This endpoint uses two LangChain PromptTemplates and a NVIDIA LLM to generate:
    1. An 8-9 line brief summary about the topic.
    2. A 2-3 line concise summary about the topic.
    Both summaries are generated in sequence and returned as a combined response.

    Args:
        topic (str): The topic to summarize, provided as a path parameter.

    Returns:
        dict: A dictionary containing the generated summaries.
    """
    # Initialize the NVIDIA LLM with desired parameters
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        task='chat',
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
    )
    # Create the first prompt template for a detailed summary
    gen_prompt = PromptTemplate(
        template="You are a very helpful asistant and you will create a brief summary about the {topic} in 8-9 lines",  # Should change the Prompt Template Later On 
        input_variables=["topic"]
    )
    # Create the second prompt template for a concise summary
    final_prompt = PromptTemplate(
        template="You are assistant and you will create a 2-3 line summary about the {topic} and also give a conclusion about the topic" , 
        input_variables=["topic"]
    )
    # Output parser to extract string output from the LLM
    parser = StrOutputParser()
    # Chain for the first summary
    llm_chain_1 = gen_prompt | llm | parser
    # Chain for the second summary
    llm_chain_2 = final_prompt | llm | parser
    # Combine both chains in sequence
    parallel_chain = llm_chain_1 | llm_chain_2
    # Run the chain asynchronously with the topic as input
    response =  parallel_chain.ainvoke({"topic": topic})
    # Return the combined summaries as a dictionary
    return ({"summary": response})