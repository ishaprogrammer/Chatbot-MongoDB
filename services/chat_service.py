# services/chat_service.py
from config import health_issues, synonyms
from data.health_issue import process_input_with_memory, product_cache
from data.general import process_with_groq, read_system_message


#------------------Routing Logic------------------------------------------------

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
