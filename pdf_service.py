#pdf_services.py
from config import health_issues
import os
import re
import PyPDF2
from services.chat_service import product_cache, check_specific_health_issue, detect_health_keywords
from data.general import read_system_message, process_with_groq
from data.health_issue import process_input_with_memory

# Replace werkzeug's secure_filename with our own implementation
def secure_filename(filename):
    """
    Create a secure version of a filename, removing potentially dangerous characters.
    """
    # Remove any path components and keep only the filename
    filename = os.path.basename(filename)
    
    # Replace spaces with underscores and remove other problematic characters
    filename = re.sub(r'[^\w\.-]', '_', filename)
    
    # Ensure filename isn't empty after sanitization
    if not filename:
        filename = 'unnamed_file'
        
    return filename

def process_pdf_content(pdf_path):
    """
    Process PDF content to determine which code path to take.
    Reuses existing detect_health_keywords and check_specific_health_issue functions.
    """
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            # Use existing functions to check for health issues
            has_specific_issue, issue = check_specific_health_issue(text)
            
            if has_specific_issue:
                # If a specific health issue is found, use health code
                return True, issue
            elif detect_health_keywords(text):
                # If general health keywords are found, still use health code
                return True, "health concern"
            else:
                # Otherwise use general code
                return False, None
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False, None

import io
from PyPDF2 import PdfReader

def handle_pdf_upload(file_object):
    """
    Process a PDF file and call the appropriate existing function
    based on the detected content. This version works in memory.
    """
    try:
        # Read the PDF content directly from the file-like object
        reader = PdfReader(file_object)
        
        # Extract text from the PDF
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Process the extracted text to determine code path
        use_health_code, keyword = process_pdf_content_from_text(text)
        
        # Generate appropriate response using existing functions
        session_id = "pdf_session"
        
        if use_health_code:
            print(f"Using health issue code for keyword: {keyword}")
            if keyword and keyword != "health concern":
                user_input = f"I have {keyword}. What products do you recommend?"
            else:
                user_input = "I have health concerns. What products do you recommend?"
            
            # Use your existing health code function
            response = process_input_with_memory(user_input, session_id, product_cache)
        else:
            print(f"Using general code path")
            user_input = "What general health products do you recommend?"
            
            # Use your existing general code function
            system_message = read_system_message("keys.txt")
            response = process_with_groq(user_input, system_message)
            
        return "Based on your reports, here are some diet tips.\n\n" + response        
    
    except Exception as e:
        return f"There was an error processing your PDF: {str(e)}"
    
def process_pdf_content_from_text(text: str):
    """
    Process the extracted text from the PDF to determine the code path.
    Replace this with your actual logic for detecting health issues or keywords.
    """
    # Example: Check for specific keywords in the text
    health_keywords = health_issues
    use_health_code = False
    keyword = None
    
    for word in health_keywords:
        if word in text.lower():
            use_health_code = True
            keyword = word
            break
    
    return use_health_code, keyword

def contains_upload_request(query):
    """
    Check if the query contains any form of upload request.
    """
    upload_patterns = [
        r'\bupload\b', 
        r'\bupload.+pdf\b', 
        r'\bupload.+report\b',
        r'\bupload.+file\b',
        r'\bshare.+pdf\b',
        r'\bshare.+report\b',
        r'\bsend.+pdf\b',
        r'\bsend.+report\b',
        r'\battach.+pdf\b',
        r'\battach.+report\b'
    ]
    
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in upload_patterns)

# Function to open file dialog using different methods
def open_file_dialog():
    """
    Opens the native system file dialog to select a PDF file.
    Tries multiple methods to ensure compatibility.
    Returns the selected file path or None if canceled.
    """
    # First, try using tkinter if available
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        print("Opening file dialog using tkinter...")
        
        # Hide the main tkinter window
        root = tk.Tk()
        root.withdraw()
        
        # Make sure the dialog appears on top
        root.attributes('-topmost', True)
        
        # Open the file dialog
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        # Close the tkinter instance
        root.destroy()
        
        if file_path:
            return file_path
        else:
            print("No file selected or dialog was cancelled.")
            return None
            
    except Exception as e:
        print(f"Error with tkinter file dialog: {e}")
        
    # If tkinter fails, try PySimpleGUI
    try:
        import PySimpleGUI as sg
        
        print("Opening file dialog using PySimpleGUI...")
        
        file_path = sg.popup_get_file(
            'Select PDF File', 
            file_types=(("PDF Files", "*.pdf"), ("All Files", "*.*")),
            no_window=True
        )
        
        if file_path:
            return file_path
        else:
            print("No file selected or dialog was cancelled.")
            return None
            
    except Exception as e:
        print(f"Error with PySimpleGUI file dialog: {e}")
    
    # If both GUI methods fail, let the user type the path
    print("\nUnable to open file dialog. Please manually enter the full path to your PDF file:")
    file_path = input("> ")
    
    if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
        return file_path
    else:
        print("Invalid file path or not a PDF file.")
        return None