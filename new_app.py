from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate
import numpy as np
import uvicorn
import warnings
import os
from difflib import get_close_matches
from datetime import datetime, timedelta

import config
from data import BASIC_CONVERSATIONS, RenFix_CATEGORIES, RenFix_keywords

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize FastAPI app
app = FastAPI(
    title="RenFix Chatbot API",
    description="Context-aware API for the RenFix customer service chatbot",
    version="2.0.0"
)

# Initialize OpenAI client
llm = OpenAI(api_key=config.OPENAI_API_KEY, temperature=0, max_tokens=100)

# Initialize MongoDB and vector store
client = MongoClient(config.MONGODB_CONN_STRING)
collection = client[config.DB_NAME][config.COLLECTION_NAME]
embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
vector_store = MongoDBAtlasVectorSearch(
    collection, 
    embeddings, 
    index_name=config.INDEX_NAME,
)

# In-memory conversation storage (use Redis/DB for production)
conversation_memory: Dict[str, List[Dict]] = {}

class ConversationTurn(BaseModel):
    user_message: str
    bot_response: str
    timestamp: datetime
    category: Optional[str] = None

class Query(BaseModel):
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = "default"  # Add session management

class ChatResponse(BaseModel):
    answer: str
    category: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to RenFix Context-Aware Chatbot API"}

def clean_old_conversations():
    """Remove conversations older than 24 hours"""
    cutoff_time = datetime.now() - timedelta(hours=24)
    for session_id in list(conversation_memory.keys()):
        conversation_memory[session_id] = [
            turn for turn in conversation_memory[session_id]
            if turn['timestamp'] > cutoff_time
        ]
        if not conversation_memory[session_id]:
            del conversation_memory[session_id]

def get_conversation_context(session_id: str, max_turns: int = 3) -> str:
    """Get recent conversation history for context"""
    if session_id not in conversation_memory:
        return ""
    
    recent_turns = conversation_memory[session_id][-max_turns:]
    context_parts = []
    
    for turn in recent_turns:
        context_parts.append(f"User: {turn['user_message']}")
        context_parts.append(f"Assistant: {turn['bot_response']}")
    
    return "\n".join(context_parts) if context_parts else ""

def save_conversation_turn(session_id: str, user_message: str, bot_response: str, category: str = None):
    """Save conversation turn to memory"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    turn = {
        'user_message': user_message,
        'bot_response': bot_response,
        'timestamp': datetime.now(),
        'category': category
    }
    
    conversation_memory[session_id].append(turn)
    
    # Keep only last 10 turns per session
    if len(conversation_memory[session_id]) > 10:
        conversation_memory[session_id] = conversation_memory[session_id][-10:]

def calculate_similarity_score(question: str, faq_question: str) -> float:
    """Calculate similarity score between user question and FAQ question"""
    try:
        q1_embedding = embeddings.embed_query(question)
        q2_embedding = embeddings.embed_query(faq_question)
        
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
    """Check if question is relevant to RenFix and return category"""
    question_lower = question.lower()
    for category, keywords in RenFix_CATEGORIES.items():
        for keyword in keywords:
            matches = get_close_matches(keyword, [question_lower], n=1, cutoff=threshold)
            if matches or keyword in question_lower:
                return True, category
    return False, ""

def is_contextually_relevant(question: str, conversation_context: str, session_id: str) -> tuple[bool, str]:
    """
    Enhanced relevance checking that considers conversation context
    """
    # First check direct keyword relevance
    is_relevant, category = is_topic_relevant(question)
    if is_relevant:
        return True, category
    
    # If no direct keywords found, check if it's a follow-up question
    if conversation_context:
        # Get the last category discussed
        if session_id in conversation_memory and conversation_memory[session_id]:
            last_turn = conversation_memory[session_id][-1]
            last_category = last_turn.get('category', '')
            
            # Check if current question could be a follow-up
            follow_up_indicators = [
                'what', 'how', 'why', 'when', 'where', 'which', 'can you', 'tell me',
                'explain', 'describe', 'show me', 'what about', 'how about',
                'benefit', 'advantage', 'feature', 'cost', 'price', 'more info',
                'details', 'it', 'this', 'that', 'them', 'they', 'what if',
                'is it', 'does it', 'will it', 'should i'
            ]
            
            question_lower = question.lower()
            has_follow_up_pattern = any(indicator in question_lower for indicator in follow_up_indicators)
            
            if has_follow_up_pattern and last_category:
                return True, last_category
            
            # Use LLM to determine if it's contextually relevant
            context_relevance = check_context_relevance_with_llm(question, conversation_context)
            if context_relevance['is_relevant']:
                return True, context_relevance['category']
    
    return False, ""

def check_context_relevance_with_llm(question: str, context: str) -> Dict[str, any]:
    """Use LLM to determine if question is contextually relevant to RenFix"""
    try:
        prompt = f"""
        Given this conversation context and current question, determine if the user's question is related to RenFix services, even if it doesn't contain specific keywords.

        Conversation Context:
        {context}

        Current Question: {question}

        RenFix provides: electrical, HVAC, plumbing, security, power solutions, registration, bookings, payments, subscriptions, profiles, reviews, notifications etc.

        Is this question relevant to RenFix based on the conversation context? Respond with only:
        RELEVANT: [category] or NOT_RELEVANT

        Examples:
        - If context mentions subscription and question asks "what benefits" → RELEVANT: subscription
        - If context mentions booking and question asks "how much does it cost" → RELEVANT: booking
        - If completely unrelated → NOT_RELEVANT
        """
        
        response = llm(prompt).strip()
        
        if response.startswith("RELEVANT:"):
            category = response.replace("RELEVANT:", "").strip()
            return {"is_relevant": True, "category": category}
        else:
            return {"is_relevant": False, "category": ""}
            
    except Exception:
        return {"is_relevant": False, "category": ""}

def get_basic_conversation_response(question: str, threshold: float = 0.85) -> str | None:
    """Returns a BASIC_CONVERSATIONS response if the question closely matches any key."""
    if question in BASIC_CONVERSATIONS:
        return BASIC_CONVERSATIONS[question]
    
    matches = get_close_matches(question, BASIC_CONVERSATIONS.keys(), n=1, cutoff=threshold)
    if matches:
        return BASIC_CONVERSATIONS[matches[0]]
    return None

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    try:
        # Clean old conversations periodically
        clean_old_conversations()
        
        question = query.question.strip().lower()
        session_id = query.session_id or "default"
        
        # Get conversation context
        conversation_context = get_conversation_context(session_id)

        # Restrict question length
        if len(question.split()) > 100:
            response = "Your question is too long. Please limit your query to 100 words or less."
            save_conversation_turn(session_id, query.question, response)
            return {"answer": response}

        # Handle basic conversations
        basic_response = get_basic_conversation_response(question)
        if basic_response:
            save_conversation_turn(session_id, query.question, basic_response, "basic")
            return {"answer": basic_response, "category": "basic"}
            
        # Enhanced relevance check with context
        is_relevant, category = is_contextually_relevant(question, conversation_context, session_id)
        
        if not is_relevant:
            response = (
                "I can only help with RenFix related questions about: "
                "Our services (electrical, HVAC, plumbing, etc.), "
                "Account & registration, "
                "Booking appointments, "
                "Payments & billing, "
                "Technical support."
            )
            save_conversation_turn(session_id, query.question, response)
            return {"answer": response}

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
                    response = content.get("answer", doc.page_content.strip())
                    save_conversation_turn(session_id, query.question, response, category)
                    return {"answer": response, "category": category}
                    
                # Generate contextual response using LLM with conversation history
                else:
                    template = """
                    You are the RenFix customer service assistant. You have access to conversation history and should provide contextual responses.

                    Conversation History:
                    {conversation_context}

                    Current User Question: {question}
                    Related FAQ: {faq_content}
                    Detected Category: {category}

                    Instructions:
                    1. Consider the conversation history when answering
                    2. If this is a follow-up question (like "what benefits", "how much", "tell me more"), reference the previous discussion
                    3. Provide specific, actionable answers in 1-2 complete sentences
                    4. For service recommendations, be specific about which RenFix service to book
                    5. For subscription questions, explain the relevant plan features
                    6. For follow-up questions about benefits/features, provide specific details
                    7. Maintain conversation flow and context

                    Service Categories:
                    - Electrical problems → Electrical Service
                    - HVAC/Cooling/Heating → HVAC Service  
                    - Plumbing/Water → Plumbing Service
                    - Subscription plans → Free (£0), Pro (£20/month), Premium (£49/month)
                    - Benefits → Explain specific features based on context

                    Provide a helpful, contextual response:
                    """
                    
                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["conversation_context", "question", "faq_content", "category"]
                    )
                    
                    response = llm(prompt.format(
                        conversation_context=conversation_context or "No previous conversation",
                        question=question,
                        faq_content=doc.page_content,
                        category=category
                    )).strip()
                    
                    save_conversation_turn(session_id, query.question, response, category)
                    return {"answer": response, "category": category}
                    
            except json.JSONDecodeError:
                response = doc.page_content.strip()
                save_conversation_turn(session_id, query.question, response, category)
                return {"answer": response, "category": category}
        
        # Fallback response
        response = "I'm sorry, I couldn't find relevant information for that query."
        save_conversation_turn(session_id, query.question, response)
        return {"answer": response}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@app.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id in conversation_memory:
        return {"session_id": session_id, "conversation": conversation_memory[session_id]}
    return {"session_id": session_id, "conversation": []}

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return {"message": f"Conversation history cleared for session {session_id}"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )