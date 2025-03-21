from pymongo import MongoClient
from groq import Groq
import os
from dotenv import load_dotenv

# ----------------- Global Variables and Initializations -----------------
load_dotenv()
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client['ChatBot']
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
conversation_history = []
final_conversation_history = []
parent_data_cache = {}
children_data_cache = {}
data_loaded = False
client = MongoClient(os.getenv("MONGODB_URI"))
db_health = client["ChatBot"]
collection = db_health["children"]
