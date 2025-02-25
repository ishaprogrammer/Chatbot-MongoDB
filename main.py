from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import os
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

conversation_history = []

class ChatInput(BaseModel):
    message: str

def read_system_message(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return "Default system message: I'm a helpful assistant."

def get_children_data(collection, parent_ids):
    children_data = []
    for parent_id in parent_ids:
        children_docs = collection.find({"Parent_id": parent_id})
        for doc in children_docs:
            if '_id' in doc:
                del doc['_id']
            children_data.append(doc)
    return children_data

def parse_values(text):
    return [value.strip() for value in text.split(',')]

def process_with_groq(user_input, system_message):
    try:
        global conversation_history

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in environment variables")
        
        groq_client = Groq(api_key=groq_key)
        
        # Connect to MongoDB
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client['ChatBot']
        parent_collection = db['parent']
        children_collection = db['children']
        
        conversation_history.append({"role": "user", "content": user_input})

        initial_messages = [
            {"role": "system", "content": system_message}
        ] + conversation_history

        response = groq_client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=initial_messages,
            temperature=0.0,
            max_tokens=1024
        )
        
        assistant_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_response})

        # Initialize category query and other conditions
        category_query = {}
        other_conditions = []
        
        if "Category:" in assistant_response:
            category_text = assistant_response.split("Category:")[1].strip().split('\n')[0]
            category_values = parse_values(category_text)
            category_conditions = []
            for value in category_values:
                category_conditions.append({
                    "Category": {
                        "$regex": value,
                        "$options": "i"
                    }
                })
            category_query = {"$or": category_conditions} if len(category_conditions) > 1 else category_conditions[0]
        
        fields = ["Medical Features:", "Tags:", "Nutritional Info:"]
        for field in fields:
            if field in assistant_response:
                field_text = assistant_response.split(field)[1].strip().split('\n')[0]
                field_values = parse_values(field_text)
                
                field_conditions = []
                for value in field_values:
                    field_conditions.append({
                        field.rstrip(":"): {
                            "$regex": value,
                            "$options": "i"
                        }
                    })
                other_conditions.extend(field_conditions)
        
        final_query = category_query
        if other_conditions:
            if category_query:
                final_query = {
                    "$and": [
                        category_query,
                        {"$or": other_conditions}
                    ]
                }
            else:
                final_query = {"$or": other_conditions}
        
        if final_query:
            matching_docs = parent_collection.find(final_query, {"Parent_id": 1, "_id": 0})
            parent_ids = [doc["Parent_id"] for doc in matching_docs]
            
            if parent_ids:
                children_data = get_children_data(children_collection, parent_ids)
                
                enhanced_system_message = """
                You are a helpful medical assistant that provides product suggestions based on the available products data.
                Reply in a way like you are talking to me if there is a casual conversation.
                Only use "₹" sign for prices.
                If user talks in Hindi, respond in Hindi but in English script.
                Do not generate  your own links or add anything to the existing links, just provide link from the data.
                Recommend minimum 1 product, maximum 3 product.
                Response should be to the point and in less words, also don't mention unnecessary info or comments.
                Please analyze the product information and provide clear recommendations including:
                1. Product names
                2. Prices
                3. Sizes
                Format the response in a clear, easy-to-read manner.

                Available Product Data:
                """
                
                for child in children_data:
                    enhanced_system_message += f"\n{child}"
                
                conversation_history.append({"role": "user", "content": user_input})

                final_messages = [
                    {"role": "system", "content": enhanced_system_message}
                ] + conversation_history

                final_response = groq_client.chat.completions.create(
                    model="qwen-2.5-32b",
                    messages=final_messages,
                    temperature=0.0,
                    max_tokens=1024
                )
                
                final_assistant_response = final_response.choices[0].message.content
                conversation_history.append({"role": "assistant", "content": final_assistant_response})

                return final_assistant_response
            else:
                return "No matching products found."
        
        return assistant_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    system_message = read_system_message("keys.txt")
    response = process_with_groq(chat_input.message, system_message)
    return {"response": response}

@app.get("/")
async def root():
    return {"message": "Chatbot API is running"}

# Handler for Vercel deployment
from mangum import Mangum
handler = Mangum(app)