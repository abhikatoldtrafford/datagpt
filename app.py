import logging
import os
import time
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import AzureOpenAI
from typing import Optional, List

app = FastAPI()

# Global context storage
global_context = {
    "assistant_id": None,
    "thread_id": None,
    "file_ids": []
}

# Azure OpenAI configuration
AZURE_ENDPOINT = "https://kb-stellar.openai.azure.com/"
AZURE_API_KEY = "bc0ba854d3644d7998a5034af62d03ce"
AZURE_API_VERSION = "2024-05-01-preview"

def create_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

async def ensure_initialized():
    """Ensure global context is initialized"""
    if not global_context["assistant_id"]:
        await initiate_chat()

@app.post("/initiate-chat")
async def initiate_chat():
    """Initialize or reset the chat session"""
    client = create_client()
    assistant_instructions = 
"""You are a data analyst specializing in spreadsheet analysis. Follow these rules:

**File Handling:**
1. If receiving Excel (.xlsx/.xls):
   - Read ALL sheets using: `df_dict = pd.read_excel(file_path, sheet_name=None)`
   - Convert each sheet to CSV named: `<original_filename>_<sheet_name>.csv` (e.g., "sales.xlsx" → "sales_Orders.csv", "sales_Clients.csv")
   - Analyze each CSV separately
   - Always reference both original file and sheet name in analysis

2. If receiving CSV:
   - Use directly for analysis
   - Preserve original filename in references

**Analysis Requirements:**
- Start with data overview: shape, columns, missing values
- Perform sheet-specific analysis for Excel files
- Compare trends across sheets when applicable
- Generate visualizations with clear source identification
- Include code snippets with explanations

**Output Formatting:**
- Begin with: "Analyzing [file.csv] / [sheet] from [file.xlsx]"
- Use markdown tables for key statistics
- Place visualizations under clear headings
- Separate analysis per sheet/file with horizontal rules"""

    try:
        # Create new assistant with code interpreter only
        assistant = client.beta.assistants.create(
            name="Data Analyst",
            instructions=assistant_instructions,
            model="gpt-4o-mini",
            tools=[{"type": "code_interpreter"}]
        )

        # Update global context
        global_context.update({
            "assistant_id": assistant.id,
            "thread_id": None,
            "file_ids": []
        })

        return JSONResponse({
            "message": "Session initialized",
            "assistant_id": assistant.id
        })

    except Exception as e:
        logging.error(f"Init error: {str(e)}")
        raise HTTPException(500, "Session initialization failed")

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads for code interpreter"""
    await ensure_initialized()
    client = create_client()
    
    try:
        # Upload file with assistants purpose
        file_content = await file.read()
        uploaded_file = client.files.create(
            file=file_content,
            purpose="assistants"
        )
        
        # Store file ID in global context
        global_context["file_ids"].append(uploaded_file.id)
        return JSONResponse({"message": "File uploaded successfully"})

    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(500, "File upload failed")

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    """Handle chat interactions with file attachments"""
    await ensure_initialized()
    client = create_client()
    
    try:
        # Create new thread with message and attachments
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "attachments": [
                        {
                            "file_id": file_id,
                            "tools": [{"type": "code_interpreter"}]
                        } for file_id in global_context["file_ids"]
                    ]
                }
            ]
        )

        # Update thread ID in context
        global_context["thread_id"] = thread.id

        # Create and poll run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=global_context["assistant_id"]
        )

        # Wait for completion
        start = time.time()
        while time.time() - start < 120:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run.status == "completed":
                break
            if run.status == "failed":
                error_msg = run.last_error.message if run.last_error else "Unknown error"
                raise HTTPException(500, detail=error_msg)
            time.sleep(2)

        # Retrieve and format response
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            order="asc"
        )

        response_content = []
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == "text":
                        response_content.append({
                            "type": "text",
                            "content": content.text.value
                        })
                    elif content.type == "image_file":
                        image_data = client.files.content(content.image_file.file_id)
                        image_bytes = image_data.read()
                        base64_image = base64.b64encode(image_bytes).decode("utf-8")
                        response_content.append({
                            "type": "image",
                            "format": "png",
                            "content": base64_image
                        })
        
        return JSONResponse({"response": response_content})

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(500, "Chat processing failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
