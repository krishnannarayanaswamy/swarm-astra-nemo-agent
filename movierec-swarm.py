from dotenv import load_dotenv
from swarm import Agent, Swarm
import os
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from nemoguardrails import LLMRails, RailsConfig
from openai import OpenAI
from typing import Dict, Any, List, Callable, Tuple
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_KEYSPACE = os.environ.get("ASTRA_DB_KEYSPACE")

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vstore = AstraDBVectorStore(
    collection_name="movies_rag",
    embedding=embedding,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    namespace=ASTRA_DB_KEYSPACE,
)

# Initialize NeMo Guardrails
config = RailsConfig.from_path("guard-config/")
rails = LLMRails(config)


def search_movies() -> str:
    """Core function to search movies in AstraVectorStore"""
    global current_query

    print("Searching Astra")
    results = vstore.similarity_search(current_query, k=5)
    response = "Here are some movie recommendations with their Title and Descriptions :\n"
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
        #title = res.metadata['title']
        response += f"- {res.page_content} \n"
        print(response)
    return response

def direct_llm_response(query: str) -> str:
    """Get response directly from LLM for non-astravectorsearch queries"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "user", "content": query}
            ]
        )
        #print(response)
        return response.choices[0].message.content or "No response generated"
    except Exception as e:
        return f"Error getting LLM response: {e}"

def is_astravectorsearch_query(response) -> bool:
    """Check if the guardrails response indicates need for astravectorsearch"""
    try:
        # Check if response contains specific indicators from our rails.co rules
        response_text = response["content"]
        isAstra = "movie database" in response_text
        #print(isAstra)
        return isAstra
    except AttributeError:
        # If we can't access the response content as expected,
        # default to treating it as a non-AstraVectorSearch query
        return False

def main():
    global current_query
    
    # Initialize Swarm client
    swarm_client = Swarm()
    
    # Initialize agent with the movie recommendation function
    agent = Agent(
        name="MovieRecommendationAgent",
        instructions="You are a helpful movie recommendation agent.",
        functions=[search_movies]
    )

    print("Welcome! You can ask me anything. Type 'exit' to quit.")
    
    while True:
        # Get user input
        user_query = input("\nYou: ").strip()
        
        if user_query.lower() == 'exit':
            print("Goodbye!")
        
        try:
            # First pass through guardrails
            guardrails_response = rails.generate(messages=[{"role": "user", "content": user_query}])
            print(guardrails_response)
            # Check if query needs astra vector search
            if is_astravectorsearch_query(guardrails_response):
                # Update current query for the search function
                print("Hello")
                current_query = user_query
                
                # Use agent for movie recommendations
                messages = [{"role": "user", "content": user_query}]
                response = swarm_client.run(agent=agent, messages=messages)
                print("Agent:", response.messages[-1]["content"])
            else:
                # Use direct LLM response for general queries
                response = direct_llm_response(user_query)
                print("Bot:", response)
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()