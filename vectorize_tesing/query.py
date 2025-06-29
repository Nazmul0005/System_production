import argparse
import params
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings
from typing import Dict

# Suppress all warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add basic conversation patterns
BASIC_CONVERSATIONS: Dict[str, str] = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! How may I assist you?",
    "hey": "Hey! What can I do for you?",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Is there anything else I can help you with?",
    "how are you": "I'm doing well, thank you! How can I assist you today?",
    "good morning": "Good morning! How may I help you?",
    "good afternoon": "Good afternoon! What can I do for you?",
    "good evening": "Good evening! How can I assist you?"
}

# Process arguments
parser = argparse.ArgumentParser(description='Atlas Vector Search Demo')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()

if args.question is None:
    # Some questions to try...
    query = "what is FixConnect?"
    query = "What are the available Services ?"
    #query = "Where is AT&T based?"
    #query = "What venues are AT&T branded?"
    #query = "How big is BofA?"
    #query = "When was the financial institution started?"
    #query = "Does the bank have an investment arm?"
    #query = "Where does the bank's revenue come from?"
    #query = "Tell me about charity."
    #query = "What buildings are BofA branded?"

else:
    query = args.question

# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection, OpenAIEmbeddings(openai_api_key=params.openai_api_key), index_name=params.index_name
)

# perform a similarity search between the embedding of the query and the embeddings of the documents
# print("\nQuery Response:")
#print("---------------")
docs = vectorStore.max_marginal_relevance_search(query, K=1)

# Contextual Compression
llm = OpenAI(
    openai_api_key=params.openai_api_key, 
    temperature=0,
    max_tokens=100,  # Limit response length
    model="gpt-3.5-turbo-instruct"  # Faster model
)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore.as_retriever()
)

# Replace the query processing part
def get_response(query: str) -> str:
    # Convert query to lowercase for matching
    query_lower = query.lower().strip()
    
    # Check if it's a basic conversation pattern
    if query_lower in BASIC_CONVERSATIONS:
        return BASIC_CONVERSATIONS[query_lower]
    
    # If not basic conversation, proceed with vector search
    try:
        compressed_docs = compression_retriever.get_relevant_documents(query)
        if compressed_docs:
            return compressed_docs[0].page_content.strip()
        return "I'm sorry, I couldn't find relevant information for that query."
    except Exception as e:
        return "I apologize, but I'm having trouble processing your request. Please try again."

# Modify the final part of the script
if __name__ == "__main__":
    response = get_response(query)
    print(response)
