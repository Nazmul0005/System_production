import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("openai_api_key")
MONGODB_CONN_STRING = os.getenv("mongodb_conn_string")
DB_NAME = os.getenv("db_name")
COLLECTION_NAME = os.getenv("collection_name")
INDEX_NAME = os.getenv("index_name")