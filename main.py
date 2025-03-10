import h5py
from pymongo import MongoClient
import random
import numpy as np
from rapidfuzz import process, fuzz
from groq import Groq
# Add to your imports at the top
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import uuid
from fastapi.middleware.cors import CORSMiddleware
import os

# Create FastAPI app
app = FastAPI(title="Health Recommendations API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create request and response models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str

# Initialize app data at startup
@app.on_event("startup")
async def startup_event():
    global recommendations, product_cache
    
    # Load recommendations from the HDF5 file
    recommendations = load_recommendations_from_h5('recommendations.h5')
    
    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client["ChatBot"]
    collection = db["children"]
    
    # Preload all product details
    product_cache = preload_all_product_details()

@app.get("/")
async def root():
    return {"message": "Health Chatbot API is running. Send POST requests to /chat endpoint."}

# Define API endpoints
@app.post("/chat", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    # Generate a session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Process the query
    response = process_input_with_memory(request.query, session_id, product_cache)
    
    return {"response": response, "session_id": session_id}


#########################################
# STEP 1: UTILITY FUNCTIONS AND SETUP
#########################################

# Dictionary mapping keywords to product IDs
keyword_to_parent_ids = {
    "combo": ["COMBARNYARD001", "COMBPROSO001", "COMBFINGER001", "COMBMILLET001"],
    "dry fruits": ["DRYFRUITS001"],
    "flour": ["FLOURGRAIN001", "BLACKWHEATFLOUR001", "LITTLEFLOUR001", "FOXTAILFLOUR001", "BARNYARDFLOUR001", "KODOFLOUR001", "MIXEDFLOUR001", "BROWNTOPFLOUR001", "RAGIFLOUR001"],
    "rice": ["BLACKRICE001", "DARKBROWNRICE001", "BROWNRICE001", "MILLETRICE001"],
    "wheat": ["WHEAT001"],
    "offer": ["COMBARNYARD001", "COMBPROSO001", "COMBFINGER001", "COMBMILLET001"],
    "black rice": ["BLACKRICE001"],
    "brown rice": ["BROWNRICE001"],
    "dark brown rice": ["DARKBROWNRICE001"],
    "millet rice": ["MILLETRICE001"],
    "black wheat flour": ["BLACKWHEATFLOUR001"],
    "little millet flour": ["LITTLEFLOUR001"],
    "foxtail millet flour": ["FOXTAILFLOUR001"],
    "Barnyard millet flour": ["BARNYARDFLOUR001"],
    "kodo millet flour": ["KODOFLOUR001"],
    "Browntop millet flour": ["BROWNTOPFLOUR001"],
    "mixed millet flour": ["MIXEDFLOUR001"],
    "ragi millet flour": ["RAGIFLOUR001"]
}

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

# Synonyms for health issues
synonyms = {
    "diabetic": "diabetes",
    "bp": "b.p"
}

# Add session-based memory
session_memory = {}

#########################################
# STEP 2: DATA LOADING FUNCTIONS
#########################################

 
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
    
    # Add parent_ids from keywords
    for keyword, parent_ids in keyword_to_parent_ids.items():
        all_parent_ids.update(parent_ids)
    
    # Fetch all products for these parent IDs at once
    for parent_id in all_parent_ids:
        products = list(collection.find(
            {"Parent_id": parent_id}, 
            {"Product Title": 1, "Price": 1, "Size": 1, "Link": 1, "Link_value": 1, "_id": 0}
        ))
        product_cache[parent_id] = products
    return product_cache

#########################################
# STEP 3: INPUT EXTRACTION FUNCTIONS
#########################################

 
def extract_keyword(user_input):
    """Extract product keyword from user input"""
    input_lower = user_input.lower()
    
    # List of specific keywords to check first (in order of priority)
    specific_keywords = [
        "black rice", "brown rice", "dark brown rice", "millet rice", "black wheat flour",
        "little millet flour", "foxtail millet flour", "Barnyard millet flour",
        "kodo millet flour", "Browntop millet flour", "mixed millet flour", "ragi millet flour",
    ]
    
    # Check for specific keywords first
    for keyword in specific_keywords:
        if keyword in input_lower:
            return keyword
    
    # Check for broader keywords if no specific keyword is found
    broader_keywords = ["rice", "flour", "wheat", "dry fruits", "combo", "offer"]
    for keyword in broader_keywords:
        if keyword in input_lower:
            return keyword
    
    return None  # No keyword found

 
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
    if "millets" in user_input_lower:
        filter_info.append("millets")
    if "products" in user_input_lower or "recommended" in user_input_lower:
        filter_info.append("products")
    return filter_info if filter_info else None

#########################################
# STEP 4: DATA FETCHING FUNCTIONS
#########################################

 
def fetch_product_details(parent_ids, product_cache):
    """Get product details from the preloaded cache"""
    product_details = {}
    for parent_id in parent_ids:
        products = product_cache.get(parent_id, [])
        if products:
            # Randomly select 2 products (or fewer if there aren't enough)
            selected_products = random.sample(products, min(1, len(products)))
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

 
def fetch_products_for_keyword(keyword, product_cache):
    """Fetch product details for the keyword's parent_ids, ensuring min 1 and max 3 products."""
    parent_ids = keyword_to_parent_ids.get(keyword, [])
    if not parent_ids:
        return {}, []  # Return empty dict and list if no parent_ids are found
    
    # Create a product_details dictionary similar to fetch_product_details
    product_details = {}
    all_selected_products = []
    
    for parent_id in parent_ids:
        if len(all_selected_products) >= 3:  # Stop if already 3 products are selected
            break
        
        products = product_cache.get(parent_id, [])
        if products:
            # Calculate how many more products can be added without exceeding the max
            remaining_slots = 3 - len(all_selected_products)
            
            # Ensure at least 1 product and at most `remaining_slots` are selected per parent ID
            sample_size = min(max(1, len(products)), remaining_slots)
            selected_products = random.sample(products, sample_size)
            
            # Format products with Link field for replace_links_with_details function
            formatted_products = []
            for product in selected_products:
                # Create a formatted product dictionary
                formatted_product = {
                    "Product Title": product["Product Title"],
                    "Price": product["Price"],
                    "Size": product["Size"],
                    "Link": product["Link"],  
                    "Link_value": product["Link_value"]
                }
                formatted_products.append(formatted_product)
                all_selected_products.append(formatted_product)
            
            product_details[parent_id] = formatted_products
        else:
            product_details[parent_id] = []
    
    # Ensure at least 1 product is returned
    if not all_selected_products:
        return {}, []  # No products found
    
    return product_details, all_selected_products

 
def get_millet_name_from_id(parent_id):
    """Convert parent_id to millet name"""
    # This is a mapping you need to define
    id_to_name = {
        "BROWNTOPMILLET001": "Brown Top Millet",
        "FOXTAILMILLET001": "Foxtail Millet",
        "FINGERMILLET001": "Finger Millet",
        "LITTLEMILLET001": "Little Millet",
        "KODOMILLET001": "Kodo Millet",
        "BARNTARDMILLET001": "Barnyard Millet",
        # Add more mappings as needed
    }
    return id_to_name.get(parent_id, parent_id)  # Fall back to ID if no mapping

#########################################
# STEP 5: RESPONSE GENERATION FUNCTIONS
#########################################

 
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

#########################################
# STEP 6: SESSION MANAGEMENT FUNCTIONS
#########################################

 
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
    
    # Basic filters
    if any(word in user_input for word in ["millet", "grain"]):
        filter_info.append("millets")
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


#########################################
# STEP 7: MAIN PROCESSING FUNCTIONS
#########################################

 
def process_keyword_request(keyword, user_input, session_id, product_cache):
    """Handle requests related to product keywords using LLM with link replacement."""
    # Fetch products for the keyword
    product_details, all_products = fetch_products_for_keyword(keyword, product_cache)
    if not all_products:
        return f"No products found for {keyword}."
    
    # Construct content for LLM introduction
    llm_intro = f"Here are the different types of {keyword.capitalize()} available:\n"
    
    # Construct the product list manually for consistency
    product_list = ""
    for index, product in enumerate(all_products, start=1):
        product_list += (
            f"{index}. {product['Product Title']}\n"
            f"   - Price: ₹{product['Price']} for {product['Size']}\n"
            f"   - Link: <{product['Link_value']}>\n\n"
        )
    
    # Combine the introduction and the product list
    final_response = llm_intro + product_list
    
    # Update session memory with the new interaction (don't update health issue)
    update_session_memory(session_id, user_input, final_response)
    
    return final_response

 
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
    """Detect the type of remedy being requested"""
    user_input_lower = user_input.lower()
    if "decoction" in user_input_lower:
        return "decoctions"
    elif "juice" in user_input_lower:
        return "juices"
    elif "oil" in user_input_lower:
        return "oils"
    return None

 
def process_input_with_memory(user_input, session_id, product_cache):
    """Main entry point for processing user input"""
    session = get_session_memory(session_id)
    
    if "customer support" in user_input.lower():
        return "You can contact on these details number 24/7:\nNumber: +918368200877\nEmail:khadar.group2021@gmail.com"
    
    elif "refund policy" in user_input.lower():
        return """Food Product Return Policy for Khadar Groups:

At Khadar Groups, we value your satisfaction and aim to provide the highest quality gourmet products. Our Food Product Return Policy is designed to address any concerns you may have regarding your purchase:

Eligibility for Returns:
Returns are accepted within 14 days of receiving your order.

Condition of Returned Items:
Returned food products must be unopened, unused, and in their original packaging.

Perishable Goods:
Perishable items, such as fresh produce or dairy, are non-returnable for safety and hygiene reasons.

Quality Concerns:
If you receive a damaged or defective food product, please contact us within 48 hours of delivery with supporting images, and we will gladly assist you."""
    
    
    # Check for remedy-specific requests first
    remedy_type = detect_remedy_type(user_input)
    if remedy_type:
        return process_remedy_request(remedy_type, user_input, session_id)
    
    # Extract keyword from user input
    keyword = extract_keyword(user_input)
    if keyword:
        return process_keyword_request(keyword, user_input, session_id, product_cache)
    
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

#########################################
# STEP 8: PROGRAM EXECUTION
#########################################

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

# Modify your main function to run the FastAPI app
if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
