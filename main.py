import h5py
from pymongo import MongoClient
import random
import numpy as np
from rapidfuzz import process, fuzz
from groq import Groq
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uvicorn
import os


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("health-chatbot")

# Initialize FastAPI app
app = FastAPI(
    title="Health Chatbot API",
    description="API for a health recommendation chatbot with millet and product suggestions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Define Pydantic models for request and response
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str

def load_recommendations_from_h5(file_path):
    recommendations = {}
    with h5py.File(file_path, 'r') as h5file:
        def load_group(group, path):
            data = {}
            for key, item in group.items():
                if isinstance(item, h5py.Group):
                    data[key] = load_group(item, path + key + '/')
                elif isinstance(item, h5py.Dataset):
                    # Convert bytes to string if necessary
                    if isinstance(item[()], bytes):
                        data[key] = item[()].decode('utf-8')
                    elif isinstance(item[()], (list, np.ndarray)):
                        data[key] = [x.decode('utf-8') if isinstance(x, bytes) else x for x in item[()]]
                    else:
                        data[key] = item[()]
            return data

        for health_issue in h5file.keys():
            recommendations[health_issue] = load_group(h5file[health_issue], health_issue + '/')
    return recommendations

def preload_all_product_details():
    """Preload all product details from MongoDB to reduce query time"""
    logger.info("Preloading all product details from MongoDB...")
    product_cache = {}
    # Get all unique parent IDs from recommendations
    all_parent_ids = set()
    for health_issue, data in recommendations.items():
        if "ids" in data:
            all_parent_ids.update(data["ids"])
    
    # Fetch all products for these parent IDs at once
    for parent_id in all_parent_ids:
        products = list(collection.find(
            {"Parent_id": parent_id}, 
            {"Product Title": 1, "Price": 1, "Size": 1, "Link": 1, "Link_value": 1, "_id": 0}
        ))
        product_cache[parent_id] = products
    
    logger.info(f"Preloaded details for {len(product_cache)} parent IDs")
    return product_cache

def fetch_product_details(parent_ids, product_cache):
    """Get product details from the preloaded cache"""
    product_details = {}
    for parent_id in parent_ids:
        products = product_cache.get(parent_id, [])
        if products:
            # Randomly select 2 products (or fewer if there aren't enough)
            selected_products = random.sample(products, min(2, len(products)))
            product_details[parent_id] = [
                {
                    "Product Title": product["Product Title"],
                    "Price": product["Price"],
                    "Size": product["Size"],
                    "Link": product["Link"],
                    "Link_value": product["Link_value"]
                }
                for product in selected_products
            ]
        else:
            product_details[parent_id] = []  # No products found for this Parent_id
    return product_details

# Connect to MongoDB
try:
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client["ChatBot"]
    collection = db["children"]
    # Test connection
    client.server_info()
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Load recommendations from the HDF5 file
try:
    recommendations = load_recommendations_from_h5('recommendations.h5')
    logger.info(f"Loaded recommendations for {len(recommendations)} health issues")
except Exception as e:
    logger.error(f"Failed to load recommendations: {e}")
    raise

# Preload all product details to avoid MongoDB queries for each request
product_cache = preload_all_product_details()

# List of health issues for matching
health_issues = [
    "dialysis", "albumin urea", "gout", "diabetes", "thyroid", "p.c.o.d", "hormonals imbalance",
    "endometriosis", "fibroid", "b.p", "heart related", "cholesterol", "triglycerides",
    "angina pectoris", "obesity / weight loss", "weight gain (underweight)", "asthma", "t.b.",
    "pneumonia", "sinusitis", "respiratory related issues", "parkinson's", "fits", "paralysis",
    "kidney stones", "gall bladder stones", "pancreas stones", "gastric problems", "acidity",
    "gerd", "eye problems", "glaucoma", "liver", "kidney", "pancreas", "hepatitis a and b",
    "nervous problems", "vertigo and migraine", "sweating in palm/feet", "snoring", "stammering",
    "tachy cardia", "after heart attack", "hole in the heart", "c4, c5", "l4, l5", "sciatica",
    "varicose veins", "varicocele", "hydrocele", "increasing platelets", "dengue fever",
    "decreasing platelets", "decreasing wbc", "infertility", "increasing sperm count",
    "constipation", "piles", "fistula", "fissures", "urine infection", "prostate (men)", "hiv",
    "skin problems", "psoriasis", "eczema dry/weeping", "vitiligo", "ichthyosis", "e.s.r",
    "urticaria", "i.b.s", "colitis", "crohn's disease", "anemia", "dental problems", "gum problems",
    "bleeding gums", "gums pain", "tooth pain", "lupus", "chikungunya", "h1 n1", "h5 n1",
    "viral fevers: malaria, typhoid", "fatty liver", "spleen", "pancreatitis", "differently abled",
    "autism", "cerebral palsy", "polio", "physically disabled", "after delivery", "during pregnancy"
]

def extract_health_issue(user_input):
    input_lower = user_input.lower()
    
    # Use fuzzy matching to find the best match from the health_issues list
    result = process.extractOne(
        input_lower, 
        health_issues, 
        scorer=fuzz.token_set_ratio,
        score_cutoff=50
    )
    
    if result:
        best_match, score, _ = result
        logger.info(f"Health issue detected: '{best_match}' with score {score}")
        if score >= 50:
            return best_match
    return None

def get_response(health_issue, filter_info=None):
    issue = health_issue.lower()
    if issue in recommendations:
        rec = recommendations[issue]
        response = ""
        
        # Add millets if available and not filtered
        if "millets" in rec and (filter_info is None or "millets" in filter_info):
            millets_info = [f"{millet} - {days}" for millet, days in rec['millets'].items() if not millet.startswith('millet_')]
            if millets_info:
                response += f"Millets: {', '.join(millets_info)}.\n"
            if "millet_note" in rec.get("millets", {}):
                response += f"  {rec['millets']['millet_note']}\n"
        
        # Fetch product details for IDs if not filtered
        if "ids" in rec and (filter_info is None or "products" in filter_info):
            product_details = fetch_product_details(rec["ids"], product_cache)
            response += "\nRecommended Products:\n"
            for parent_id, products in product_details.items():
                if products:
                    millet_name = get_millet_name_from_id(parent_id)
                    response += f"{millet_name}:\n"
                    for product in products:
                        response += f" {product['Link']}\n"
        
        # Add decoctions if available and specifically asked for
        if "decoctions" in rec and (filter_info is not None and "decoctions" in filter_info):
            response += f"Decoctions: {', '.join(rec['decoctions'])}.\n"
            if "decoction_instruction" in rec:
                response += f"  {rec['decoction_instruction']}.\n"
        
        # Add juices and additional notes from the "more" field if specifically asked for
        if "more" in rec and (filter_info is not None and "more" in filter_info):
            if "juices" in rec["more"]:
                response += f"Juices: {', '.join(rec['more']['juices'])}.\n"
                if "juice_instruction" in rec["more"]:
                    response += f"  {rec['more']['juice_instruction']}\n"
            if "additional_notes" in rec["more"]:
                response += f"Additional Notes: {rec['more']['additional_notes']}\n"
        
        return response.rstrip()
    return "Sorry, I don't have information on that health issue."

def get_millet_name_from_id(parent_id):
    # This is a mapping you need to define
    id_to_name = {
        "BROWNTOPMILLET001": "Brown Top Millet",
        "FOXTAILMILLET001": "Foxtail Millet",
        "FINGERMILLET001": "Finger Millet",
        "LITTLEMILLET001": "Little Millet",
        "KODOMILLET001": "Kodo Millet",
        "BARNTARDMILLET001": "Barnyard Millet",        # Add more mappings as needed
    }
    return id_to_name.get(parent_id, parent_id)  # Fall back to ID if no mapping

def determine_filter_info(user_input):
    user_input_lower = user_input.lower()
    filter_info = []
    if "decoctions" in user_input_lower:
        filter_info.append("decoctions")
    if "more" in user_input_lower or "additional" in user_input_lower:
        filter_info.append("more")
    if "millets" in user_input_lower:
        filter_info.append("millets")
    if "products" in user_input_lower or "recommended" in user_input_lower:
        filter_info.append("products")
    return filter_info if filter_info else None

def send_to_groq(content):
    try:
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        groq_client = Groq(api_key=api_key)
        
        # Define the prompt for Groq
        prompt = f"""Based on the following information, generate a concise and helpful response. For recommended products.
    Use numbered list format for the millet types, with each millet on its own line followed by a colon.
    Do not generate Product Title, Price, Size, Link_value, or any additional links, URLs or information. Use only the information provided.
    Do not include any parent_id.

        Here is the information:
        {content}
        """     
        
        # Call Groq API
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            max_tokens=500,
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        return f"Sorry, I encountered an error while processing your request. {str(e)}"

# Add session-based memory
session_memory = {}

def get_session_memory(session_id):
    if session_id not in session_memory:
        session_memory[session_id] = {"history": []}
    return session_memory[session_id]

def update_session_memory(session_id, user_input, bot_response, health_issue):
    session = get_session_memory(session_id)
    session["history"].append({"user": user_input, "bot": bot_response, "health_issue": health_issue})

def process_input_with_memory(user_input, session_id):
    session = get_session_memory(session_id)
    
    # Extract health issue from user input
    health_issue = extract_health_issue(user_input)
    if not health_issue and session["history"]:  # If no health issue is mentioned, use the last one
        health_issue = session["history"][-1].get("health_issue")
    
    if health_issue:
        # Determine what info to filter based on user input
        filter_info = determine_filter_info(user_input)
        content = get_response(health_issue, filter_info)
        
        # Get product details for replacement
        product_details = {}
        if health_issue.lower() in recommendations and "ids" in recommendations[health_issue.lower()]:
            product_details = fetch_product_details(recommendations[health_issue.lower()]["ids"], product_cache)
        
        # Send the content to Groq for final response
        llm_response = send_to_groq(content)
        
        # Replace Links with Product Title, Price, and Size in the final response
        final_response = replace_links_with_details(llm_response, product_details)
        
        # Update session memory with the new interaction
        update_session_memory(session_id, user_input, final_response, health_issue)
        return final_response, health_issue
    return "Please specify a health issue from the list.", None

def replace_links_with_details(llm_response, product_details):
    # First, remove any ID patterns like (BROWNTOPMILLET001) that might appear
    import re
    llm_response = re.sub(r'\([A-Z0-9]+\)', '', llm_response)
    
    # Now replace links with product details
    for _, products in product_details.items():
        for product in products:
            # Format product details according to requested format
            formatted_details = f"{product['Product Title']}\n Price: ₹{product['Price']} for {product['Size']} [{product['Link_value']}]\n"
            
            # Remove any remaining "Link" values
            # Replace the Link with formatted details
            llm_response = llm_response.replace(product["Link"], formatted_details)
            
    llm_response = re.sub(r'Link-\d+', '', llm_response)

    # Clean up any formatting issues
    llm_response = llm_response.replace("  ", " ")
    llm_response = llm_response.replace("\n\n-", "\n-")
    
    return llm_response

# FastAPI endpoints
@app.get("/")
async def root():
    return {"message": "Health Chatbot API is running. Send POST requests to /chat endpoint."}

# FastAPI endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Process the user's message
    response, health_issue = process_input_with_memory(request.message, session_id)
    
    # Return only the response as required
    return ChatResponse(
        response=response
    )

# Health issues endpoint
@app.get("/health_issues")
async def get_health_issues():
    """Return the list of supported health issues"""
    return {"health_issues": sorted(health_issues)}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")