from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ----------------- Imports -----------------
import h5py
from pymongo import MongoClient
import random
import numpy as np
from rapidfuzz import process, fuzz
from groq import Groq
from config import health_issues, session_memory, synonyms, refund_policy
import os
from dotenv import load_dotenv
from functools import lru_cache
import time
import re

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

#------------------FastAPI setup--------------------------------

# Define FastAPI app
app = FastAPI()

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    
@app.get("/")
async def root():
    return {"message": "Health Chatbot API is running. Send POST requests to /chat endpoint."}

# Define the /chat endpoint
@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Endpoint to handle user queries.
    """
    try:
        # Use the provided session_id or default to "default_session" if not provided
        session_id = request.session_id if request.session_id else "default_session"

        # Process the user input using the existing logic
        response = process_user_input(request.query, session_id)

        # Return the response along with the session_id
        return QueryResponse(response=response, session_id=session_id)
    except Exception as e:
        # Handle any errors and return a 500 status code with the error message
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- Health Issue Code Functions -----------------
def load_recommendations_from_h5(file_path):
    """Load health recommendations from HDF5 file"""
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
    product_cache = {}
    
    # Get all unique parent IDs from recommendations and keywords
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
    return product_cache

 
def extract_health_issue(user_input):
    """Extract health issue from user input using fuzzy matching"""
    input_lower = user_input.lower()
    
    # Replace input with synonym if found
    if input_lower in synonyms:
        input_lower = synonyms[input_lower]

    # Use fuzzy matching with partial_ratio
    result = process.extractOne(
        input_lower,
        health_issues,
        scorer=fuzz.partial_ratio,
        score_cutoff=50
    )

    if result:
        best_match, score, _ = result
        return best_match if score >= 50 else None
    return None

def determine_filter_info(user_input):
    """Determine what information to filter based on user input"""
    user_input_lower = user_input.lower()
    filter_info = []
    if "decoctions" in user_input_lower:
        filter_info.append("decoctions")
    if "more" in user_input_lower or "additional" in user_input_lower:
        filter_info.append("more")
    if "products" in user_input_lower or "recommended" in user_input_lower:
        filter_info.append("products")
    return filter_info if filter_info else None

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
            print("no products found")
            product_details[parent_id] = []  # No products found for this Parent_id
    return product_details


 
def get_millet_name_from_id(parent_id):
    """Convert parent_id to millet name"""
    id_to_name = {
        "BROWNTOPMILLET001": "Brown Top Millet",
        "FOXTAILMILLET001": "Foxtail Millet",
        "FINGERMILLET001": "Finger Millet",
        "LITTLEMILLET001": "Little Millet",
        "KODOMILLET001": "Kodo Millet",
        "BARNTARDMILLET001": "Barnyard Millet",
    }
    return id_to_name.get(parent_id, parent_id)  # Fall back to ID if no mapping

 
def get_response(health_issue, filter_info=None):
    """Generate initial response for health issue"""
    
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
        
        # Add juices and additional notes if specifically asked for
        if "more" in rec:
            # Juices
            if filter_info is not None and "juices" in filter_info and "juices" in rec["more"]:
                response += f"Juices: {', '.join(rec['more']['juices'])}.\n"
                if "juice_instruction" in rec["more"]:
                    response += f"  {rec['more']['juice_instruction']}\n"
            
            # Oils
            if filter_info is not None and "oils" in filter_info and "oils" in rec["more"]:
                response += f"Oils: {', '.join(rec['more']['oils'])}.\n"
                if "oil_instruction" in rec["more"]:
                    response += f"  {rec['more']['oil_instruction']}\n"
            
            # Additional notes for general "more" filter
            if filter_info is not None and "more" in filter_info and "additional_notes" in rec["more"]:
                response += f"Additional Notes: {rec['more']['additional_notes']}\n"
        
        return response.rstrip()
    return "Sorry, I don't have information on that health issue."

 
def get_specific_remedy_info(health_issue, remedy_type):
    """Extract specific remedy information for a health issue"""
    issue = health_issue.lower()
    if issue not in recommendations:
        return f"No information about {remedy_type} for {health_issue}."
    
    rec = recommendations[issue]
    response = ""
    
    # Handle decoctions
    if remedy_type == "decoctions":
        if "decoctions" in rec and rec["decoctions"]:
            response += f"Recommended decoctions for {health_issue}:\n"
            response += f"{', '.join(rec['decoctions'])}.\n"
            if "decoction_instruction" in rec:
                response += f"Instructions: {rec['decoction_instruction']}.\n"
        else:
            response = f"No decoctions are needed for {health_issue}."
    
    # Handle juices
    elif remedy_type == "juices":
        if "more" in rec and "juices" in rec["more"] and rec["more"]["juices"]:
            response += f"Recommended juices for {health_issue}:\n"
            response += f"{', '.join(rec['more']['juices'])}.\n"
            if "juice_instruction" in rec["more"]:
                response += f"Instructions: {rec['more']['juice_instruction']}.\n"
        else:
            response = f"No juices are needed for {health_issue}."
    
    # Handle oils
    elif remedy_type == "oils":
        if "more" in rec and "oils" in rec["more"] and rec["more"]["oils"]:
            response += f"Recommended oils for {health_issue}:\n"
            response += f"{', '.join(rec['more']['oils'])}.\n"
            if "oil_instruction" in rec["more"]:
                response += f"Instructions: {rec['more']['oil_instruction']}.\n"
        else:
            response = f"No oils are needed for {health_issue}."
    
    else:
        response = f"No {remedy_type} information available for {health_issue}."
    
    return response

def send_to_groq(content):
    """Send content to Groq LLM for natural language generation"""
    # Initialize Groq client
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
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

 
def replace_links_with_details(llm_response, product_details):
    """Replace link placeholders with detailed product information"""
    # First, remove any ID patterns like (BROWNTOPMILLET001) that might appear
    import re
    llm_response = re.sub(r'\([A-Z0-9]+\)', '', llm_response)
    
    # Now replace links with product details
    for _, products in product_details.items():
        for product in products:
            # Format product details including Link_value
            formatted_details = f"{product['Product Title']}\n Price: ₹{product['Price']} for {product['Size']} [{product['Link_value']}]\n"
            
            # Replace the Link with formatted details
            llm_response = llm_response.replace(product["Link"], formatted_details)
            
    llm_response = re.sub(r'Link-\d+', '', llm_response)

    # Clean up any formatting issues
    llm_response = llm_response.replace("  ", " ")
    llm_response = llm_response.replace("\n\n-", "\n-")
    
    return llm_response

def get_session_memory(session_id):
    """Get or create session memory for a user"""
    if session_id not in session_memory:
        session_memory[session_id] = {
            "history": [],
            "last_health_issue": None
        }
    return session_memory[session_id]

 
def update_session_memory(session_id, user_input, bot_response, health_issue=None):
    """Update session memory with new interaction"""
    session = get_session_memory(session_id)
    session["history"].append({"user": user_input, "bot": bot_response})
    
    # Only update last_health_issue if a new health issue is provided and it's not None
    if health_issue:
        session["last_health_issue"] = health_issue
        # Log the last health issue for debugging

 
def determine_filter_info(user_input):
    """Determine what information to filter based on user input"""
    user_input = user_input.lower()
    
    filter_info = []
    
    if any(word in user_input for word in ["product", "buy", "purchase"]):
        filter_info.append("products")
    
    # Treatment-specific filters
    if "decoction" in user_input:
        filter_info.append("decoctions")
    if "juice" in user_input:
        filter_info.append("juices")
    if "oil" in user_input:
        filter_info.append("oils")
    if any(word in user_input for word in ["more", "detail", "additional"]):
        filter_info.append("more")
    
    return filter_info if filter_info else None
 
def process_health_issue_request(health_issue, user_input, session_id, product_cache):
    """Handle requests related to health issues"""
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
    
    # Update session memory with the new interaction and the health issue
    update_session_memory(session_id, user_input, final_response, health_issue)
    
    return final_response

 
def process_remedy_request(remedy_type, user_input, session_id):
    """Process requests for decoctions, juices, or oils from last health issue"""
    session = get_session_memory(session_id)
    last_health_issue = session.get("last_health_issue")
    
    if not last_health_issue:
        return "Please first ask about a specific health issue before requesting remedies."
    
    # Get detailed information about the specific remedy for the last health issue
    remedy_info = get_specific_remedy_info(last_health_issue, remedy_type)
    
    # Check if the response indicates no remedies are needed
    if remedy_info.startswith(f"No {remedy_type} are needed"):
        # Return directly without sending to Groq to avoid generating generic content
        update_session_memory(session_id, user_input, remedy_info)
        return remedy_info
    
    # Format the response to be sent to Groq
    formatted_info = f"For {last_health_issue}, {remedy_info}"
    
    # Send to Groq for natural language formatting
    llm_response = send_to_groq(formatted_info)
    
    # Check if the LLM response appears to be generic and doesn't contain specific remedies
    # This helps prevent the LLM from making up information
    if not any(word in llm_response.lower() for word in [remedy_type[:-1], "remedy", "recommend"]):
        # If the response seems generic, revert to our direct response
        llm_response = remedy_info
    
    # Update session memory with the interaction (but don't change last_health_issue)
    update_session_memory(session_id, user_input, llm_response)
    
    return llm_response

 
def detect_remedy_type(user_input):
    """Detect the type of remedy being requested."""
    user_input_lower = user_input.lower()
    if "decoction" in user_input_lower or "decoctions" in user_input_lower:
        return "decoctions"
    elif "juice" in user_input_lower or "juices" in user_input_lower:
        return "juices"
    elif "oil" in user_input_lower or "oils" in user_input_lower:
        return "oils"
    return None

 
def process_input_with_memory(user_input, session_id, product_cache):
    """Main entry point for processing user input"""
    session = get_session_memory(session_id)
    
    if "customer support" in user_input.lower():
        return "You can contact on these details number 24/7:\nNumber: +918368200877\nEmail:khadar.group2021@gmail.com"
    
    elif "refund policy" in user_input.lower():
        return refund_policy
    
    
    # Check for remedy-specific requests first
    remedy_type = detect_remedy_type(user_input)
    if remedy_type:
        return process_remedy_request(remedy_type, user_input, session_id)
    
    # Extract health issue from user input
    health_issue = extract_health_issue(user_input)
    
    # If health issue found, process it
    if health_issue:
        return process_health_issue_request(health_issue, user_input, session_id, product_cache)
    
    # If no health issue is found and there's a last health issue in memory
    # process as follow-up question about the last health issue
    if session.get("last_health_issue"):
        # Create filter info based on the query
        filter_info = determine_filter_info(user_input)
        # Get a response based on the last health issue
        response = f"Based on your previous question about {session['last_health_issue']}, here's what I can tell you:\n\n"
        response += process_health_issue_request(session["last_health_issue"], user_input, session_id, product_cache)
        return response
    
    return "Please specify a health issue or a product category from the list."

# Load recommendations from the HDF5 file
recommendations = load_recommendations_from_h5('recommendations.h5')

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["ChatBot"]
collection = db["children"]

# Preload all product details
product_cache = preload_all_product_details()

# Add overall timing for the whole process
def process_with_timing(query, session_id):
    result = process_input_with_memory(query, session_id, product_cache)
    return result

# ----------------- General Query Code Functions -----------------
def load_all_data():
    """Load all needed data into memory once"""
    global parent_data_cache, children_data_cache, data_loaded
    
    if data_loaded:
        return
    
    # Fetch all parent documents with projection to reduce memory usage
    parent_collection = db['parent']
    parents = list(parent_collection.find({}, {
        "_id": 0, 
        "Parent_id": 1, 
        "Category": 1, 
        "Medical Features": 1, 
        "Tags": 1, 
        "Nutritional Info": 1
    }))
    
    # Index parents by ID for fast lookup
    for parent in parents:
        parent_id = parent["Parent_id"]
        parent_data_cache[parent_id] = parent
    
    # Fetch all children documents with needed fields
    children_collection = db['children']
    children = list(children_collection.find({}, {"_id": 0}))
    
    # Group children by parent_id
    for child in children:
        parent_id = child.get("Parent_id")
        if parent_id:
            if parent_id not in children_data_cache:
                children_data_cache[parent_id] = []
            children_data_cache[parent_id].append(child)
    
    data_loaded = True

@lru_cache(maxsize=1)
def read_system_message(file_path):
    """Cache the system message to avoid repeated file reads"""
    start_time = time.time()
    try:
        with open(file_path, 'r') as file:
            result = file.read().strip()
    except FileNotFoundError:
        result = "Default system message: I'm a helpful assistant."
    end_time = time.time()
    return result

def parse_values(text):
    """Parse comma-separated values"""
    return [value.strip() for value in text.split(',')]

def filter_parents_in_memory(assistant_response):
    matching_parent_ids = []
    category_values = []
    medical_features = []
    tags = []
    nutritional_info = []
    
    # Extract search criteria from assistant response
    if "Category:" in assistant_response:
        category_text = assistant_response.split("Category:")[1].strip().split('\n')[0]
        category_values = parse_values(category_text)
    
    if "Medical Features:" in assistant_response:
        medical_text = assistant_response.split("Medical Features:")[1].strip().split('\n')[0]
        medical_features = parse_values(medical_text)
    
    if "Tags:" in assistant_response:
        tags_text = assistant_response.split("Tags:")[1].strip().split('\n')[0]
        tags = parse_values(tags_text)
    
    if "Nutritional Info:" in assistant_response:
        nutritional_text = assistant_response.split("Nutritional Info:")[1].strip().split('\n')[0]
        nutritional_info = parse_values(nutritional_text)
    
    # Filter parents based on criteria
    for parent_id, parent in parent_data_cache.items():
        # Check if parent matches any category
        category_match = False
        if category_values:
            parent_category = parent.get("Category", "")
            for category in category_values:
                if category.lower() in parent_category.lower():
                    category_match = True
                    break
        else:
            category_match = True  # No category filter specified
        
        if not category_match:
            continue
        
        # Check for additional criteria
        other_match = True
        
        # Check medical features
        if medical_features and other_match:
            parent_medical = parent.get("Medical Features", "")
            med_match = False
            for feature in medical_features:
                if feature.lower() in parent_medical.lower():
                    med_match = True
                    break
            other_match = med_match
        
        # Check tags
        if tags and other_match:
            parent_tags = parent.get("Tags", "")
            tags_match = False
            for tag in tags:
                if tag.lower() in parent_tags.lower():
                    tags_match = True
                    break
            other_match = tags_match
        
        # Check nutritional info
        if nutritional_info and other_match:
            parent_nutrition = parent.get("Nutritional Info", "")
            nutrition_match = False
            for info in nutritional_info:
                if info.lower() in parent_nutrition.lower():
                    nutrition_match = True
                    break
            other_match = nutrition_match
        
        # If parent matches all criteria, add to results
        if category_match and other_match:
            matching_parent_ids.append(parent_id)
            
            # Limit to 6 parents
            if len(matching_parent_ids) >= 6:
                break
    
    return matching_parent_ids

def get_children_for_parents(parent_ids, limit=6):
    
    children_data = []
    link_map = {}  # Dictionary to store Link -> Link_value mapping
    
    for parent_id in parent_ids:
        if parent_id in children_data_cache:
            children_data.extend(children_data_cache[parent_id])
            
            # Limit total children
            if len(children_data) >= limit:
                children_data = children_data[:limit]
                break
    
    # Clean the data (remove images, links) but save the link values for later
    cleaned_children_data = []
    for item in children_data:
        # Save the link value before removing it
        if "Link" in item and "Link_value" in item:
            link_map[item["Link"]] = item["Link_value"]
        
        # Make a copy to avoid modifying the original
        cleaned_item = item.copy()
        
        # Explicitly remove any "Images" key if it exists
        if "Images" in cleaned_item:
            del cleaned_item["Images"]
        
        # Explicitly remove the "Link_value" key if it exists
        if "Link_value" in cleaned_item:
            del cleaned_item["Link_value"]
        
        # Look for any fields containing image URLs
        for key in list(cleaned_item.keys()):
            if isinstance(cleaned_item[key], str) and (
                "http" in cleaned_item[key] and any(ext in cleaned_item[key].lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])
            ):
                # If the field appears to be solely an image URL, remove it
                if key.lower() in ['image', 'images', 'img', 'thumbnail', 'photo']:
                    del cleaned_item[key]
        
        cleaned_children_data.append(cleaned_item)
    return cleaned_children_data, link_map

def replace_link_placeholders(response_text, link_map):
    """Replace [Link-X] placeholders with actual link values, and handle entire response being Link-X"""
    # Check if the entire response is Link-X
    entire_link_pattern = r'^Link-(\d+)$'
    entire_match = re.match(entire_link_pattern, response_text.strip())
    if entire_match:
        link_id = f"Link-{entire_match.group(1)}"
        if link_id in link_map:
            return link_map[link_id]
        else:
            return response_text  # If link not found, return original
    
    # Otherwise, replace [Link-X] placeholders within the text
    link_pattern = r'\[Link-(\d+)\]'
    matches = re.findall(link_pattern, response_text)
    
    modified_response = response_text
    for match in matches:
        link_id = f"Link-{match}"
        placeholder = f"[{link_id}]"
        if link_id in link_map:
            actual_link = link_map[link_id]
            modified_response = modified_response.replace(placeholder, actual_link)
    
    return modified_response

def process_with_groq(user_input, system_message):
    """Process user input with Groq model and return response"""
    total_start_time = time.time()
    try:
        global conversation_history, final_conversation_history
        
        # Ensure data is loaded
        if not data_loaded:
            load_all_data()
        if not system_message:
            system_message = read_system_message("keys.txt")
        
        # Add user input to both conversation histories
        conversation_history.append({"role": "user", "content": user_input})
        final_conversation_history.append({"role": "user", "content": user_input})
        
        initial_messages = [
            {"role": "system", "content": system_message}
        ] + conversation_history
        
         # Debug: Log the initial messages being sent to the LLM
        print("Debug: Initial Messages Sent to LLM:")
        for msg in initial_messages:
            print(f"{msg['role'].capitalize()}: {msg['content']}")
        
        response = groq_client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=initial_messages,
            temperature=0.0,
            max_tokens=1024
        )
        
        assistant_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_response})
        
        matching_parent_ids = filter_parents_in_memory(assistant_response)
        
        # Get children data from memory and link map
        children_data, link_map = get_children_for_parents(matching_parent_ids) if matching_parent_ids else ([], {})
        
        enhanced_system_message = """You are a helpful medical assistant that provides product suggestions based on the available products data but if no data is provided then answer from your side DONT USE below FORMAT.
            Only use "₹" sign for prices.
            If user talks in Hindi, respond in Hindi but in English script.
            Do not create your own links, just provide link from the data.
            Recommend min 1 and max 3 products from the provided data(if provided otherwise dont recommend any product).
            Response should be to the point and concise, also don't mention unnecessary info or comments.
            Please analyze the product information and provide clear recommendations in a proper format including:
            1. Product names
               - Prices
               - Sizes
               - [Link]
               
            Available Product Data:
            """
        
        # Prepare product data string efficiently
        product_data_str = "\n".join(str(child) for child in children_data)
        enhanced_system_message += product_data_str
        
        modified_user_input = f"{user_input} (response should be in proper format and easy to read. Respond like you're assisting)"
        conversation_history.append({"role": "user", "content": modified_user_input})

        # Use final_conversation_history for the second API call
        final_messages = [
            {"role": "system", "content": enhanced_system_message}
        ] + final_conversation_history
        
         # Debug: Log the final messages being sent to the LLM
        print("Debug: Final Messages Sent to LLM:")
        for msg in final_messages:
            print(f"{msg['role'].capitalize()}: {msg['content']}")
        
        final_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-specdec",
            messages=final_messages,
            temperature=0.0,
            max_tokens=1024
        )
        
        l_assistant_response = final_response.choices[0].message.content
        
        final_assistant_response = replace_link_placeholders(l_assistant_response, link_map)
        
        conversation_history.append({"role": "assistant", "content": final_assistant_response})
        # Add the raw LLM response to final_conversation_history
        final_conversation_history.append({"role": "assistant", "content": l_assistant_response})

        return final_assistant_response
    
    except Exception as e:
        total_end_time = time.time()
        print(f"Error occurred. Total processing time: {total_end_time - total_start_time:.4f} seconds")
        raise Exception(f"Error: {str(e)}")
    
def chat_endpoint(message):
    system_message = read_system_message("keys.txt")
    response = process_with_groq(message, system_message)
    return {"response": response}

recommendations = load_recommendations_from_h5('recommendations.h5')
product_cache = preload_all_product_details()
load_all_data()
# ----------------- Routing Logic -----------------
def detect_health_keywords(user_input):
    """Detect health-related keywords or remedy keywords in the user's input."""
    input_lower = user_input.lower()
    
    # Check for health issues and their synonyms
    for keyword in health_issues:
        if keyword.lower() in input_lower:
            return True
    
    # Check synonyms and map them to actual health issues
    for synonym, actual_issue in synonyms.items():
        if synonym.lower() in input_lower:
            return True
    
    # Check for remedy keywords
    remedy_keywords = ["decoction", "decoctions", "juice", "juices", "oil", "oils"]
    for keyword in remedy_keywords:
        if keyword in input_lower:
            return True
    
    # If no health issue or remedy keyword is found, return False
    return False

def check_specific_health_issue(user_input):
    """Check if user input contains a specific health issue from the health_issues list or its synonyms."""
    input_lower = user_input.lower()
    
    # First check for exact matches in health_issues
    for issue in health_issues:
        if issue.lower() in input_lower:
            return True, issue
    
    # Check for synonyms and return the actual health issue if found
    for synonym, actual_issue in synonyms.items():
        if synonym.lower() in input_lower:
            return True, actual_issue
    
    return False, None

def process_user_input(user_input, session_id):
    """Main entry point for processing user input."""
    # First, check if input contains a specific health issue or synonym
    has_specific_issue, issue = check_specific_health_issue(user_input)
    
    if has_specific_issue:
        # If we found a specific health issue or its synonym, process with health issue code
        return process_input_with_memory(user_input, session_id, product_cache)
    elif detect_health_keywords(user_input):
        # If no specific health issue was found but there are general health keywords,
        # still process with health issue code
        return process_input_with_memory(user_input, session_id, product_cache)
    else:
        # If no health-related content is found, proceed with general query code
        system_message = read_system_message("keys.txt")
        return process_with_groq(user_input, system_message)

# ----------------- Main Loop -----------------
# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
