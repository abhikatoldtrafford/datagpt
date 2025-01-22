import logging
import os
import time
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import AzureOpenAI
from typing import Optional

app = FastAPI()

# Global context storage
global_context = {
    "assistant_id": None,
    "thread_id": None,
    "vector_store_id": None
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
    if not all(global_context.values()):
        await initiate_chat()

@app.post("/initiate-chat")
async def initiate_chat(file: UploadFile = File(None)):
    """Initialize or reset the chat session"""
    client = create_client()
    
    try:
        # Create new resources
        vector_store = client.beta.vector_stores.create(name="main_store")
        assistant = client.beta.assistants.create(
            name="Data Analyst",
            instructions="Analyze data using code interpreter and file search",
            model="gpt-4o-mini",
            tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
            tool_resources={
                "file_search": {"vector_store_ids": [vector_store.id]},
                "code_interpreter": {"file_ids": []}
            }
        )
        thread = client.beta.threads.create()

        # Update global context
        global_context.update({
            "assistant_id": assistant.id,
            "thread_id": thread.id,
            "vector_store_id": vector_store.id
        })

        # Handle initial file upload
        if file:
            await upload_file(file)

        return JSONResponse({
            "message": "Session initialized",
            "session_id": thread.id
        })

    except Exception as e:
        logging.error(f"Init error: {str(e)}")
        raise HTTPException(500, "Session initialization failed")

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads for current session"""
    await ensure_initialized()
    client = create_client()
    
    try:
        # Save file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Upload to vector store
        with open(file_path, "rb") as f:
            client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=global_context["vector_store_id"],
                files=[f]
            )

        return JSONResponse({"message": "File uploaded successfully"})

    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(500, "File upload failed")

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    """Handle chat interactions with image support"""
    await ensure_initialized()
    client = create_client()
    
    try:
        # Add message to thread
        client.beta.threads.messages.create(
            thread_id=global_context["thread_id"],
            role="user",
            content=prompt
        )

        # Create and poll run
        run = client.beta.threads.runs.create(
            thread_id=global_context["thread_id"],
            assistant_id=global_context["assistant_id"]
        )

        # Wait for completion
        start = time.time()
        while time.time() - start < 120:
            run = client.beta.threads.runs.retrieve(
                thread_id=global_context["thread_id"],
                run_id=run.id
            )
            if run.status == "completed":
                break
            time.sleep(2)

        # Retrieve and format response
        messages = client.beta.threads.messages.list(
            thread_id=global_context["thread_id"],
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
                        try:
                            # Get image bytes
                            image_data = client.files.content(content.image_file.file_id)
                            image_bytes = image_data.read()
                            
                            # Convert to base64
                            base64_image = base64.b64encode(image_bytes).decode("utf-8")
                            
                            response_content.append({
                                "type": "image",
                                "format": "png",
                                "content": base64_image
                            })
                        except Exception as e:
                            logging.error(f"Error processing image: {e}")
                            continue

        return JSONResponse({"response": response_content})

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(500, "Chat processing failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
