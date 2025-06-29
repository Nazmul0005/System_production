from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate  # Add this import
import numpy as np  # Add this import if not present
import uvicorn
import warnings
import os
from difflib import get_close_matches
from fastapi.middleware.cors import CORSMiddleware

import config
from data import BASIC_CONVERSATIONS, RenFix_CATEGORIES, RenFix_keywords
# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize FastAPI app
app = FastAPI(
    title="RenFix Chatbot API",
    description="API for the RenFix customer service chatbot",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize OpenAI client
llm = OpenAI(api_key=config.OPENAI_API_KEY, temperature=0, max_tokens=30)

# Initialize MongoDB and vector store
client = MongoClient(config.MONGODB_CONN_STRING)
collection = client[config.DB_NAME][config.COLLECTION_NAME]
embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
vector_store = MongoDBAtlasVectorSearch(
    collection, 
    embeddings, 
    index_name=config.INDEX_NAME,
)

# Basic conversation patterns



class Query(BaseModel):
    question: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Welcome to RenFix Chatbot API"}

def calculate_similarity_score(question: str, faq_question: str) -> float:
    """Calculate similarity score between user question and FAQ question"""
    try:
        # Get embeddings for both questions
        q1_embedding = embeddings.embed_query(question)
        q2_embedding = embeddings.embed_query(faq_question)
        
        # Calculate cosine similarity
        similarity = np.dot(q1_embedding, q2_embedding) / (
            np.linalg.norm(q1_embedding) * np.linalg.norm(q2_embedding)
        )
        return float(similarity)
    except Exception:
        return 0.0

def is_similar_to_keywords(word: str, keywords: list, threshold: float = 0.8) -> bool:
    """Check if word is similar to any keyword using fuzzy matching"""
    for keyword in keywords:
        matches = get_close_matches(word, [keyword], n=1, cutoff=threshold)
        if matches:
            return True
    return False

def is_topic_relevant(question: str, threshold: float = 0.8) -> tuple[bool, str]:
    """
    Check if question is relevant to RenFix and return category
    Returns: (is_relevant, category)
    """
    question_lower = question.lower()
    for category, keywords in RenFix_CATEGORIES.items():
        for keyword in keywords:
            # Fuzzy match the keyword with the question
            matches = get_close_matches(keyword, [question_lower], n=1, cutoff=threshold)
            if matches or keyword in question_lower:
                return True, category
    return False, ""

def get_basic_conversation_response(question: str, threshold: float = 0.85) -> str | None:
    """
    Returns a BASIC_CONVERSATIONS response if the question closely matches any key.
    """
    if question in BASIC_CONVERSATIONS:
        return BASIC_CONVERSATIONS[question]
    # Fuzzy match
    matches = get_close_matches(question, BASIC_CONVERSATIONS.keys(), n=1, cutoff=threshold)
    if matches:
        return BASIC_CONVERSATIONS[matches[0]]
    return None

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    try:
        question = query.question.strip().lower()

        # Restrict question to 100 words before embedding
        if len(question.split()) > 100:
            return {"answer": "Your question is too long. Please limit your query to 100 words or less."}

        # Handle basic conversations with fuzzy logic
        basic_response = get_basic_conversation_response(question)
        if basic_response:
            return {"answer": basic_response}
            
        # Check topic relevance
        is_relevant, category = is_topic_relevant(question)
        if not is_relevant:
            return {
                "answer": "I can only help with RenFix related questions about:" " Our services (electrical, HVAC, plumbing, etc.),"
                         " Account & registration,"
                         " Booking appointments,"
                         " Payments & billing,"
                         " Technical support."
            }

        # Check if question is related to RenFix topics
  
        # Split question into words and check each word for similarity
        question_words = question.split()
        is_relevant = any(
            is_similar_to_keywords(word, RenFix_keywords)
            for word in question_words
        )
        
        if not is_relevant:
            return {"answer": "I can only help with RenFix related questions. Please ask about our services, booking, registration, or technical support or try to rephrase your message."}

        # Get similar FAQ entries
        docs = vector_store.similarity_search_with_score(question, k=1)
        
        if docs:
            doc, score = docs[0]
            try:
                content = json.loads(doc.page_content)
                faq_question = content.get("question", "")
                similarity_score = calculate_similarity_score(question, faq_question)
                
                # If similarity is high (>0.9), return FAQ answer directly
                if similarity_score > 0.9:
                    return {"answer": content.get("answer", doc.page_content.strip())}
                    
                # If similarity is lower, generate contextual response using LLM
                else:
                    template = """
                    You are the RenFix customer service assistant. Analyze the user's question and provide a specific, relevant response:

                    User Question: {question}
                    Related FAQ: {faq_content}

                    Instructions:
                    1. Your response must be concise and short with proper meaning. 
                    One complete sentence that addresses all user needs without cutting off.You do not finish with incomplete sentence. Finish in one sentence with proper meaning. No explnation. Direct answer which provides the solutiona or suggestion.
                    2. Respond in ONE SENTENCE only
                    3. Use commas and conjunctions to combine multiple service recommendations
                    4. Be direct and specific about which service to book
                    5. Start with action words like "Book", "Schedule", "Use" when recommending services
                    6. No extra explanations or pleasantries
                    7. For new users → Explain registration process
                    8. For service issues → Match to specific service:
                       - Electrical problems → Electrical Service
                        (e.g., wiring faults, flickering lights, circuit breaker trips)

                       - HVAC/Cooling/Heating → HVAC Service
                        (e.g., AC not cooling, heater failure, thermostat issues)

                       - Plumbing/Water → Plumbing Service
                        (e.g., leaks, clogged drains, low water pressure)

                       - Security Systems → Security Service
                        (e.g., alarm failures, CCTV issues, smart lock problems)

                       - Power Issues → Power Solutions
                        (e.g., outages, generator failures, voltage fluctuations)

                       - Appliance Repairs → Appliance Maintenance
                        (e.g., fridge not cooling, washing machine leaks, oven malfunctions)

                       - Pest Control → Extermination Service
                        (e.g., rodents, insects, termite infestations)

                       - Carpentry/Woodwork → Carpentry Service
                        (e.g., broken doors, furniture repair, shelving installation)

                       - Painting & Decorating → Painting Service
                        (e.g., wall cracks, peeling paint, interior/exterior repainting)

                       - Landscaping/Gardening → Landscaping Service
                        (e.g., lawn care, tree trimming, irrigation problems)

                       - Roofing/Waterproofing → Roofing Service
                        (e.g., leaks, shingle damage, gutter cleaning)

                       - Cleaning/Sanitization → Cleaning Service
                        (e.g., deep cleaning, post-construction cleanup, disinfection)

                       - IT/Tech Support → IT Services
                        (e.g., network issues, hardware repairs, software troubleshooting)

                       - Waste Management → Junk Removal/Recycling
                        (e.g., trash pickup, bulk waste disposal, recycling queries)

                       - Elevator/Escalator → Elevator Maintenance
                        (e.g., stuck elevators, unusual noises, door malfunctions)

                       - Fire Safety → Fire Protection Service
                        (e.g., extinguisher refills, smoke detector checks, sprinkler systems)
                    9. For general inquiries → Explain relevant RenFix feature
                    10. Keep response to few words if possible and if it is not possible to answer the query in few words then use 1 sentence maximum
                    11. Be specific and action-oriented
                    12. Don't default to electrical service

                    Response should focus on the user's specific need or question.
                    """
                    
                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["question", "faq_content"]
                    )
                    
                    response = llm(prompt.format(
                        question=question,
                        faq_content=doc.page_content
                    ))
                    
                    return {"answer": response.strip()}
                    
            except json.JSONDecodeError:
                return {"answer": doc.page_content.strip()}
        
        return {"answer": "I'm sorry, I couldn't find relevant information for that query."}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # Changed from main:app to app:app
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )