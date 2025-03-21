#routes.py

from controllers.pdf_service import handle_pdf_upload, contains_upload_request
from fastapi import UploadFile, File, Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import io
from controllers.chat_service import process_user_input

router = APIRouter()

# Session storage (in-memory for this example)
sessions = {}

class ChatRequest(BaseModel):
    query: Optional[str] = None
    session_id: Optional[str] = None
    file: Optional[UploadFile] = None

class ChatResponse(BaseModel):
    message: str
    session_id: str
    expecting_upload: bool = False

@router.post("/chat", response_model=ChatResponse)
async def chat(
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    session_id: str = Form(...)  # Require session_id from the frontend
):
    # Initialize session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {
            "initial_question_answered": False,
            "expecting_upload": False,
            "pdf_processed": False
        }
    
    session = sessions[session_id]
    
    # Handle file upload if provided
    if file:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a PDF"
            )
        
        try:
            # Read the file content into memory
            file_content = await file.read()
            
            # Create a file-like object from the content
            pdf_file = io.BytesIO(file_content)
            
            # Process the PDF in memory
            pdf_response = handle_pdf_upload(pdf_file)
            
            # Update session state
            session["expecting_upload"] = False
            session["initial_question_answered"] = True
            session["pdf_processed"] = True  # Set the PDF processed flag
            
            # Ignore text input if both file and text are provided
            return ChatResponse(
                message=pdf_response,
                session_id=session_id
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred: {str(e)}"
            )
    
    # Handle text input if no file is uploaded
    if query:
        # Handle exit command
        if query.lower() in ["exit", "quit"]:
            return ChatResponse(message="Goodbye!", session_id=session_id)
        
        # Handle the initial question if not yet answered
        if not session["initial_question_answered"]:
            if query.lower() == "continue":
                session["initial_question_answered"] = True
                return ChatResponse(
                    message="How can I help you today?",
                    session_id=session_id
                )
            elif query.lower() == "upload" or contains_upload_request(query):
                session["expecting_upload"] = True
                return ChatResponse(
                    message="Please upload your PDF",
                    session_id=session_id,
                    expecting_upload=True
                )
            else:
                # If they didn't say continue or upload, treat it as continue
                session["initial_question_answered"] = True
                response = process_user_input(query, session_id)
                return ChatResponse(
                    message=f"I'll proceed without health reports. {response}",
                    session_id=session_id
                )
        
        # For subsequent queries, check if they want to upload a file anytime
        if contains_upload_request(query) and not session["expecting_upload"]:
            session["expecting_upload"] = True
            return ChatResponse(
                message="Please upload your PDF",
                session_id=session_id,
                expecting_upload=True
            )
        else:
            # Regular query processing
            response = process_user_input(query, session_id)
            return ChatResponse(
                message=response,
                session_id=session_id
            )
    
    # If neither file nor text input is provided
    raise HTTPException(
        status_code=400,
        detail="No file or text input provided"
    )
    
@router.post("/clear")
async def clear_chat(session_id: str = Form(...)):
    if session_id in sessions:
        del sessions[session_id]
        return JSONResponse(
            content={"message": "Chat session cleared successfully.", "session_id": session_id}
        )
    return JSONResponse(
        content={"message": "Session ID not found.", "session_id": session_id},
        status_code=404
    )
        
