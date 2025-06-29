# https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader  # Changed import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import params

# Step 1: Load
file_path = "D:/System_Chatbot/vectorize_tesing/final.json"
loader = JSONLoader(
    file_path=file_path,
    jq_schema='.[]',  # Adjust this based on your JSON structure
    text_content=False
)

data = loader.load()

# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                               "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(data)
print('Split into ' + str(len(docs)) + ' docs')

# Step 3: Embed
# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# Reset w/out deleting the Search Index 
collection.delete_many({})

# Insert the documents in MongoDB Atlas with their embedding
# https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/mongodb_atlas.py
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.index_name
)