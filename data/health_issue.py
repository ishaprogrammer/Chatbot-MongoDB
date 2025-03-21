#health_issue code
# ----------------- Imports -----------------
import h5py
from pymongo import MongoClient
import numpy as np
from rapidfuzz import process, fuzz
from groq import Groq
from config import health_issues, session_memory, synonyms, refund_policy
import os


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
        # print(f"Parent_id: {parent_id}, Products found: {len(products)}")
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
    # print(f"Attempting to fetch details for these parent_ids: {parent_ids}")

    product_details = {}
    for parent_id in parent_ids:
        products = product_cache.get(parent_id, [])
        if products:
            selected_products = [products[0]]
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
    Do not include any parent_id.
    Do not remove any information.
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
        model="llama-3.3-70b-specdec",
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
    
     # Add usage and benefits information at the end, only if products were recommended
    if any(products for _, products in product_details.items()):
        llm_response += "\n\nUsage:\n- Eat for breakfast or snacks. Cook like rice or make porridge, upma, or rotis.\n\n"
        llm_response += "Benefits:\n- Rich in protein, boosts energy, and supports digestion."
    
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
