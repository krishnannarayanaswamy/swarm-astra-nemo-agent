A simple Multi-agent RAG that combines DataStax Astra's vector database capabilities with OpenAI's language models to provide intelligent movie recommendations. The system uses NeMo Guardrails for query classification and input and output validation, and DataStax Astra agent for querying the database with a ANN query (vector search). You can extend it to also do exact keyword search and semantic search as part of the same agent.

Architecture
The application consists of several key components:

DataStax Astra Database: Stores movie data and handles vector-based similarity searches and exact keyword search
OpenAI Integration: Provides natural language understanding and generation
NeMo Guardrails: Manages conversation flows and ensures appropriate responses
Swarm Agent System: Coordinates between different components to handle user queries

Installation

Clone the repository:

git clone https://github.com/krishnannarayanaswamy/swarm-astra-nemo-agent.git

Install the dependencies:

pip install -r requirements.txt

Set up environment variables in a .env file:

OPENAI_API_KEY=your_openai_api_key

ASTRA_DB_APPLICATION_TOKEN=your_astra_token

ASTRA_DB_API_ENDPOINT=your_astra_endpoint

ASTRA_DB_KEYSPACE=your_astra_keyspace


How It Works
The application follows this flow:

Query Processing:

User input is first processed through NeMo Guardrails
Guardrails classify the query into different types (movie recommendations, general questions, etc.)
Query Routing:

Movie-related queries are directed to the DataStax Astra database
General queries are handled directly by the OpenAI LLM
Off-topic queries (politics, stock market) are filtered with appropriate responses
Movie Recommendations:

For movie queries, the system searches the DataStax Astra database
Results are ranked by relevance
Top 10 recommendations are returned with relevance scores

Usage
Run the script:

python movierec-swarm.py

The system will start an interactive session where you can:

Ask for movie recommendations
Ask general questions
Type 'exit' to quit
Extending the Application
You can expand this application in several ways:

Database Enhancement:

Add more movie metadata (genres, actors, directors)
Implement more sophisticated vector similarity searches
Add user ratings and viewing history
Query Processing:

Extend NeMo Guardrails rules in nemo-configs/rails.co
Add new conversation flows
Implement more sophisticated query understanding
Response Generation:

Add personalization based on user preferences
Implement more detailed movie descriptions
Add multi-turn conversations about movies
New Features:

Add collaborative filtering
Implement user profiles
Add movie clustering
Integrate with external movie APIs
To extend the application:

Adding New Guardrails:

Edit nemo-configs/rails.co to add new patterns and flows
Update config.yml if needed for model configurations
Database Modifications:

Add new columns to the movies table in DataStax Astra
Update the search_movies() function in movierec-swarm.py
Agent Enhancement:

Add new functions to the MovieRecommendationAgent
Implement new swarm patterns for complex queries
API Integration:

Add new API clients in the main script
Implement new data sources for enriched recommendations