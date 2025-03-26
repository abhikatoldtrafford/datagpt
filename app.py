import logging
from fastapi import FastAPI, Request, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AzureOpenAI
from typing import Optional, List, Dict, Any
import os
import datetime
import time
import base64
import mimetypes
import uuid

app = FastAPI()

# Azure OpenAI client configuration
AZURE_ENDPOINT = "https://kb-stellar.openai.azure.com/"
AZURE_API_KEY = "bc0ba854d3644d7998a5034af62d03ce"
AZURE_API_VERSION = "2024-05-01-preview"

# Configure file storage directory
FILE_STORAGE_DIR = "/tmp/assistant_files"
os.makedirs(FILE_STORAGE_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

# Accept and ignore additional parameters
async def ignore_additional_params(request: Request):
    form_data = await request.form()
    return {k: v for k, v in form_data.items()}

# Helper function to check if file is an image
def is_image_file(filename: str, content_type: Optional[str] = None) -> bool:
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in image_extensions or (content_type and content_type.startswith('image/'))

# Helper function to check if file is CSV
def is_csv_file(filename: str, content_type: Optional[str] = None) -> bool:
    return filename.lower().endswith('.csv') or (content_type and content_type == 'text/csv')

# Helper function to check if file is Excel
def is_excel_file(filename: str, content_type: Optional[str] = None) -> bool:
    excel_extensions = ['.xls', '.xlsx', '.xlsm', '.xlsb']
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in excel_extensions or (content_type and content_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'])

async def image_analysis(client, image_data: bytes, filename: str, prompt: Optional[str] = None) -> str:
    """Analyzes an image using Azure OpenAI vision capabilities and returns the analysis text."""
    try:
        ext = os.path.splitext(filename)[1].lower()
        b64_img = base64.b64encode(image_data).decode("utf-8")
        # Default to jpeg if extension can't be determined
        mime = f"image/{ext[1:]}" if ext and ext[1:] in ['jpg', 'jpeg', 'png', 'gif', 'webp'] else "image/jpeg"
        data_url = f"data:{mime};base64,{b64_img}"
        
        default_prompt = (
            "Analyze this image and provide a thorough summary including all elements. "
            "If there's any text visible, include all the textual content. Describe:"
        )
        combined_prompt = f"{default_prompt} {prompt}" if prompt else default_prompt
        
        # Use the chat completions API to analyze the image
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": combined_prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
                ]
            }],
            max_tokens=500
        )
        
        analysis_text = response.choices[0].message.content
        return analysis_text
        
    except Exception as e:
        logging.error(f"Image analysis error: {e}")
        return f"Error analyzing image: {str(e)}"
        
# Helper function to update user persona context
async def update_context(client, thread_id, context):
    """Updates the user persona context in a thread by adding a special message."""
    if not context:
        return
        
    try:
        # Get existing messages to check for previous context
        messages = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=20  # Check recent messages
        )
        
        # Look for previous context messages to avoid duplication
        has_previous_context = False
        for msg in messages.data:
            if hasattr(msg, 'metadata') and msg.metadata.get('type') == 'user_persona_context':
                # Delete previous context message to replace it
                try:
                    client.beta.threads.messages.delete(
                        thread_id=thread_id,
                        message_id=msg.id
                    )
                except Exception as e:
                    logging.error(f"Error deleting previous context message: {e}")
                    # Continue even if delete fails
                has_previous_context = True
                break
        
        # Add new context message
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=f"USER PERSONA CONTEXT: {context}",
            metadata={"type": "user_persona_context"}
        )
        
        logging.info(f"Updated user persona context in thread {thread_id}")
    except Exception as e:
        logging.error(f"Error updating context: {e}")
        # Continue the flow even if context update fails

# Helper function to add file metadata message to thread
async def add_file_metadata_to_thread(client, thread_id, file_info):
    """Adds a message with file metadata to the thread for assistant awareness."""
    if not thread_id or not file_info:
        return
        
    try:
        file_type = file_info.get('type', 'unknown')
        file_name = file_info.get('name', 'unnamed')
        file_path = file_info.get('path', '')
        file_id = file_info.get('file_id', '')
        
        metadata_message = f"""
FILE UPLOADED - METADATA:
- Name: {file_name}
- Type: {file_type}
- ID: {file_id}
- Path: {file_path}
"""
        if file_type == 'image':
            metadata_message += f"- Analysis: {file_info.get('analysis', 'No analysis available')}"
        
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=metadata_message,
            metadata={"type": "file_metadata"}
        )
        
        logging.info(f"Added file metadata for {file_name} to thread {thread_id}")
    except Exception as e:
        logging.error(f"Error adding file metadata to thread: {e}")

@app.post("/initiate-chat")
async def initiate_chat(request: Request, **kwargs):
    """
    Initiates the assistant and session and optionally uploads a file to its vector store, 
    all in one go.
    """
    client = create_client()

    # Parse the form data
    form = await request.form()
    file = form.get("file", None)
    context = form.get("context", None)  # New: Get optional context parameter

    # Create a vector store up front
    vector_store = client.beta.vector_stores.create(name="demo")

    # Enhanced system prompt with file handling instructions
    system_prompt = '''
        You are a highly skilled Product Management AI Assistant and Co-Pilot. Your primary responsibilities include generating comprehensive Product Requirements Documents (PRDs) and providing insightful answers to a wide range of product-related queries. You seamlessly integrate information from uploaded files and your extensive knowledge base to deliver contextually relevant and actionable insights.

        ### **Primary Tasks:**

        1. **Generate Product Requirements Documents (PRDs):**
        - **Trigger:** When the user explicitly requests a PRD.
        - **Structure:**
            - **Product Manager:** [Use the user's name if available; otherwise, leave blank]
            - **Product Name:** [Derived from user input or uploaded files]
            - **Product Vision:** [Extracted from user input or uploaded files]
            - **Customer Problem:** [Identified from user input or uploaded files]
            - **Personas:** [Based on user input; generate if not provided]
            - **Date:** [Current date]
        
        - **Sections to Include:**
            - **Executive Summary:** Deliver a concise overview by synthesizing information from the user and your knowledge base.
            - **Goals & Objectives:** Enumerate 2-4 specific, measurable goals and objectives.
            - **Key Features:** Highlight key features that align with the goals and executive summary.
            - **Functional Requirements:** Detail 3-5 functional requirements in clear bullet points.
            - **Non-Functional Requirements:** Outline 3-5 non-functional requirements in bullet points.
            - **Use Case Requirements:** Describe 3-5 use cases in bullet points, illustrating how users will interact with the product.
            - **Milestones:** Define 3-5 key milestones with expected timelines in bullet points.
            - **Risks:** Identify 3-5 potential risks and mitigation strategies in bullet points.

        2. **Answer Generic Product Management Questions:**
        - **Scope:** Respond to a broad range of product management queries, including strategy, market analysis, feature prioritization, user feedback interpretation, and more.
        - **Methodology:**
            - Use the file_search tool to find pertinent information within uploaded files.
            - Leverage your comprehensive knowledge base to provide thorough and insightful answers.
            - If a question falls outside the scope of the provided files and your expertise, default to a general GPT-4 response without referencing the files.
            - Maintain a balance between technical detail and accessibility, ensuring responses are understandable yet informative.

        ### **File Handling:**

        1. **File Awareness:**
            - Always maintain awareness of files that have been uploaded.
            - If the user references a file by name, understand which file they're referring to.
            - Remember file metadata including file type, name, and purpose.

        2. **Data File Processing:**
            - For CSV files, use code_interpreter to analyze data, generate statistics, and create visualizations.
            - For Excel files, process each sheet separately and maintain awareness of all sheets.
            - Always reference the original filename and sheet name (for Excel) in your analysis.

        3. **Image Processing:**
            - For image files, refer to the analysis provided with the image.
            - Incorporate relevant information from the image into your responses.

        4. **Analysis Requirements:**
            - Start with data overview: shape, columns, missing values
            - Perform sheet-specific analysis for Excel files
            - Compare trends across sheets when applicable
            - Generate visualizations with clear source identification
            - Include code snippets with explanations

        5. **Output Formatting:**
            - Begin with: "Analyzing [file.csv] / [sheet] from [file.xlsx]"
            - Use markdown tables for key statistics
            - Place visualizations under clear headings
            - Separate analysis per sheet/file with horizontal rules

        ### **Behavioral Guidelines:**

        - **Contextual Awareness:**
        - Always consider the context provided by the uploaded files and previous interactions.
        - Adapt your responses based on the specific needs and preferences of the user.

        - **Proactive Insight Generation:**
        - Go beyond surface-level answers by providing deep insights, trends, and actionable recommendations.
        - Anticipate potential follow-up questions and address them preemptively where appropriate.

        - **Professional Tone:**
        - Maintain a professional, clear, and concise communication style.
        - Ensure all interactions are respectful, objective, and goal-oriented.

        - **Continuous Improvement:**
        - Learn from each interaction to enhance future responses.
        - Seek feedback when necessary to better align with the user's expectations and requirements.

        ### **Important Notes:**

        - **Tool Utilization:**
        - Always evaluate whether the file_search tool or code_interpreter can enhance the quality of your response.
        - Use code_interpreter for data analysis tasks, especially with CSV and Excel files.
        
        - **Data Privacy:**
        - Handle all uploaded files and user data with the utmost confidentiality and in compliance with relevant data protection standards.

        - **Assumption Handling:**
        - Clearly indicate when you are making assumptions due to missing information.
        - Provide rationales for your assumptions to maintain transparency.

        - **Error Handling:**
        - Gracefully manage any errors or uncertainties by informing the user and seeking clarification when necessary.

        By adhering to these guidelines, you will function as an effective Product Management AI Assistant, delivering high-quality PRDs and insightful answers that closely mimic the expertise of a seasoned product manager.
        '''

    # Always include file_search and code_interpreter tools
    assistant_tools = [{"type": "code_interpreter"}, {"type": "file_search"}]
    assistant_tool_resources = {"file_search": {"vector_store_ids": [vector_store.id]}}

    # Create the assistant
    try:
        assistant = client.beta.assistants.create(
            name="demo_new_abhik",
            model="gpt-4o-mini",
            instructions=system_prompt,
            tools=assistant_tools,
            tool_resources=assistant_tool_resources,
        )
    except BaseException as e:
        logging.info(f"An error occurred while creating the assistant: {e}")
        raise HTTPException(status_code=400, detail="An error occurred while creating assistant")

    logging.info(f'Assistant created {assistant.id}')

    # Create a thread
    try:
        thread = client.beta.threads.create()
    except BaseException as e:
        logging.info(f"An error occurred while creating the thread: {e}")
        raise HTTPException(status_code=400, detail="An error occurred while creating the thread")

    logging.info(f"Thread created: {thread.id}")

    # If context is provided, add it as user persona context
    if context:
        try:
            await update_context(client, thread.id, context)
        except BaseException as e:
            logging.info(f"An error occurred while adding context to the thread: {e}")
            # Don't fail the entire request if just adding context fails

    # If a file is provided, upload it now
    if file:
        filename = file.filename
        file_content = await file.read()
        file_path = os.path.join(FILE_STORAGE_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(file_content)
            
        # Check file type
        file_type = None
        file_id = None
        if is_csv_file(filename) or is_excel_file(filename):
            # For CSV or Excel, upload to assistant files for code_interpreter
            try:
                with open(file_path, "rb") as file_stream:
                    uploaded_file = client.files.create(
                        file=file_stream,
                        purpose="assistants"
                    )
                    file_id = uploaded_file.id
                    file_type = "csv" if is_csv_file(filename) else "excel"
                    
                    # Update assistant to use this file with code_interpreter
                    client.beta.assistants.update(
                        assistant_id=assistant.id,
                        tool_resources={
                            "code_interpreter": {"file_ids": [file_id]},
                            "file_search": {"vector_store_ids": [vector_store.id]}
                        }
                    )
                    logging.info(f"File uploaded for code_interpreter: {file_id}")
                    
                    # Add file metadata to thread
                    file_info = {
                        "name": filename,
                        "type": file_type,
                        "path": file_path,
                        "file_id": file_id
                    }
                    await add_file_metadata_to_thread(client, thread.id, file_info)
            except Exception as e:
                logging.error(f"Error uploading file for code_interpreter: {e}")
        elif is_image_file(filename):
            # For images, analyze and add to thread
            try:
                analysis_text = await image_analysis(client, file_content, filename, None)
                
                # Save analysis text to a file
                analysis_file_path = os.path.join(FILE_STORAGE_DIR, f"{filename}_analysis.txt")
                with open(analysis_file_path, 'w') as f:
                    f.write(analysis_text)
                
                # Add image analysis to thread
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=f"Image Analysis for {filename}: {analysis_text}"
                )
                
                # Add file metadata to thread
                file_info = {
                    "name": filename,
                    "type": "image",
                    "path": file_path,
                    "analysis": analysis_text
                }
                await add_file_metadata_to_thread(client, thread.id, file_info)
                
                logging.info(f"Image analyzed and added to thread: {filename}")
            except Exception as e:
                logging.error(f"Error analyzing image: {e}")
        else:
            # For other file types, upload to vector store
            try:
                with open(file_path, "rb") as file_stream:
                    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store.id, 
                        files=[file_stream]
                    )
                    logging.info(f"File uploaded to vector store: status={file_batch.status}, count={file_batch.file_counts}")
                    
                    # Add file metadata to thread
                    file_info = {
                        "name": filename,
                        "type": "document",
                        "path": file_path,
                        "vector_store_id": vector_store.id
                    }
                    await add_file_metadata_to_thread(client, thread.id, file_info)
            except Exception as e:
                logging.error(f"Error uploading file to vector store: {e}")

    res = {
        "assistant": assistant.id,
        "session": thread.id,
        "vector_store": vector_store.id
    }

    return JSONResponse(res, media_type="application/json", status_code=200)


@app.post("/co-pilot")
async def co_pilot(request: Request, **kwargs):
    """
    Handles co-pilot creation or updates with optional file upload and system prompt.
    """
    client = create_client()
    
    # Parse the form data
    form = await request.form()
    file = form.get("file", None)
    system_prompt = form.get("system_prompt", None)
    context = form.get("context", None)  # New: Get optional context parameter

    # Attempt to get the assistant & vector store from the form
    assistant_id = form.get("assistant", None)
    vector_store_id = form.get("vector_store", None)
    thread_id = form.get("session", None)

    # Enhanced base prompt with file handling instructions
    base_prompt = """
    You are a product management AI assistant, a product co-pilot.
    
    ### File Handling:
    1. If receiving Excel (.xlsx/.xls):
       - Read ALL sheets using: `df_dict = pd.read_excel(file_path, sheet_name=None)`
       - Convert each sheet to CSV named: `<original_filename>_<sheet_name>.csv` 
       - Analyze each CSV separately
       - Always reference both original file and sheet name in analysis

    2. If receiving CSV:
       - Use directly for analysis
       - Preserve original filename in references

    3. If receiving images:
       - Reference the image analysis provided
       - Incorporate relevant details from the image in your responses

    ### Analysis Requirements:
    - Start with data overview: shape, columns, missing values
    - Perform sheet-specific analysis for Excel files
    - Compare trends across sheets when applicable
    - Generate visualizations with clear source identification
    - Include code snippets with explanations

    ### Output Formatting:
    - Begin with: "Analyzing [file.csv] / [sheet] from [file.xlsx]"
    - Use markdown tables for key statistics
    - Place visualizations under clear headings
    - Separate analysis per sheet/file with horizontal rules
    """

    # If no assistant, create one
    if not assistant_id:
        if not vector_store_id:
            vector_store = client.beta.vector_stores.create(name="demo")
            vector_store_id = vector_store.id
        
        instructions = base_prompt if not system_prompt else f"{base_prompt}\n\n{system_prompt}"
        
        assistant = client.beta.assistants.create(
            name="demo_co_pilot",
            model="gpt-4o-mini",
            instructions=instructions,
            tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
        assistant_id = assistant.id
    else:
        # If user gave an assistant, update instructions if needed
        if system_prompt:
            updated_instructions = f"{base_prompt}\n\n{system_prompt}"
            client.beta.assistants.update(
                assistant_id=assistant_id,
                instructions=updated_instructions,
            )
        # If no vector_store, check existing or create new
        if not vector_store_id:
            assistant_obj = client.beta.assistants.retrieve(assistant_id=assistant_id)
            file_search_resource = getattr(assistant_obj.tool_resources, "file_search", None)
            existing_stores = (
                file_search_resource.vector_store_ids
                if (file_search_resource and hasattr(file_search_resource, "vector_store_ids"))
                else []
            )
            if existing_stores:
                vector_store_id = existing_stores[0]
            else:
                vector_store = client.beta.vector_stores.create(name="demo")
                vector_store_id = vector_store.id
                existing_tools = assistant_obj.tools if assistant_obj.tools else []
                
                # Make sure both code_interpreter and file_search are included
                if not any(t["type"] == "file_search" for t in existing_tools):
                    existing_tools.append({"type": "file_search"})
                if not any(t["type"] == "code_interpreter" for t in existing_tools):
                    existing_tools.append({"type": "code_interpreter"})
                
                client.beta.assistants.update(
                    assistant_id=assistant_id,
                    tools=existing_tools,
                    tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
                )

    # Handle file upload if present
    if file:
        filename = file.filename
        file_content = await file.read()
        file_path = os.path.join(FILE_STORAGE_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(file_content)
            
        # Check file type
        file_type = None
        file_id = None
        if is_csv_file(filename) or is_excel_file(filename):
            # For CSV or Excel, upload to assistant files for code_interpreter
            try:
                with open(file_path, "rb") as file_stream:
                    uploaded_file = client.files.create(
                        file=file_stream,
                        purpose="assistants"
                    )
                    file_id = uploaded_file.id
                    file_type = "csv" if is_csv_file(filename) else "excel"
                    
                    # Get current code_interpreter file_ids if any
                    assistant_obj = client.beta.assistants.retrieve(assistant_id=assistant_id)
                    code_interpreter_resource = getattr(assistant_obj.tool_resources, "code_interpreter", None)
                    existing_file_ids = (
                        code_interpreter_resource.file_ids
                        if (code_interpreter_resource and hasattr(code_interpreter_resource, "file_ids"))
                        else []
                    )
                    
                    # Add new file_id to existing ones
                    updated_file_ids = existing_file_ids + [file_id]
                    
                    # Update assistant to use this file with code_interpreter
                    tool_resources = {
                        "file_search": {"vector_store_ids": [vector_store_id]},
                        "code_interpreter": {"file_ids": updated_file_ids}
                    }
                    
                    client.beta.assistants.update(
                        assistant_id=assistant_id,
                        tool_resources=tool_resources
                    )
                    logging.info(f"File uploaded for code_interpreter: {file_id}")
                    
                    # Add file metadata to thread if thread exists
                    if thread_id:
                        file_info = {
                            "name": filename,
                            "type": file_type,
                            "path": file_path,
                            "file_id": file_id
                        }
                        await add_file_metadata_to_thread(client, thread_id, file_info)
            except Exception as e:
                logging.error(f"Error uploading file for code_interpreter: {e}")
        elif is_image_file(filename):
            # For images, analyze and add to thread if thread exists
            if thread_id:
                try:
                    analysis_text = await image_analysis(client, file_content, filename, None)
                    
                    # Save analysis text to a file
                    analysis_file_path = os.path.join(FILE_STORAGE_DIR, f"{filename}_analysis.txt")
                    with open(analysis_file_path, 'w') as f:
                        f.write(analysis_text)
                    
                    # Add image analysis to thread
                    client.beta.threads.messages.create(
                        thread_id=thread_id,
                        role="user",
                        content=f"Image Analysis for {filename}: {analysis_text}"
                    )
                    
                    # Add file metadata to thread
                    file_info = {
                        "name": filename,
                        "type": "image",
                        "path": file_path,
                        "analysis": analysis_text
                    }
                    await add_file_metadata_to_thread(client, thread_id, file_info)
                    
                    logging.info(f"Image analyzed and added to thread: {filename}")
                except Exception as e:
                    logging.error(f"Error analyzing image: {e}")
        else:
            # For other file types, upload to vector store
            try:
                with open(file_path, "rb") as file_stream:
                    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store_id,
                        files=[file_stream]
                    )
                    logging.info(f"File uploaded to vector store: status={file_batch.status}, count={file_batch.file_counts}")
                    
                    # Add file metadata to thread if thread exists
                    if thread_id:
                        file_info = {
                            "name": filename,
                            "type": "document",
                            "path": file_path,
                            "vector_store_id": vector_store_id
                        }
                        await add_file_metadata_to_thread(client, thread_id, file_info)
            except Exception as e:
                logging.error(f"Error uploading file to vector store: {e}")

    # If context provided and thread exists, update context
    if context and thread_id:
        try:
            await update_context(client, thread_id, context)
        except BaseException as e:
            logging.info(f"An error occurred while adding context to the thread: {e}")
            # Don't fail the entire request if just adding context fails

    return JSONResponse(
        {
            "message": "Assistant updated successfully.",
            "assistant": assistant_id,
            "vector_store": vector_store_id,
        }
    )


@app.post("/upload-file")
async def upload_file(file: UploadFile = Form(...), assistant: str = Form(...), **kwargs):
    """
    Uploads a file and associates it with the given assistant.
    Handles CSV, Excel, and image files specially.
    """
    client = create_client()
    # Get context if provided (in kwargs or form data)
    context = kwargs.get("context", None)
    thread_id = kwargs.get("session", None)
    prompt = kwargs.get("prompt", None)  # Optional prompt for image analysis

    try:
        # Save the uploaded file locally and get the data
        file_content = await file.read()
        file_path = os.path.join(FILE_STORAGE_DIR, file.filename)
        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)
            
        # Determine file type
        is_img = is_image_file(file.filename, file.content_type)
        is_csv = is_csv_file(file.filename, file.content_type)
        is_excel = is_excel_file(file.filename, file.content_type)
        
        # Retrieve the assistant
        assistant_obj = client.beta.assistants.retrieve(assistant_id=assistant)
        
        # Check if there's a file_search resource for vector store
        file_search_resource = getattr(assistant_obj.tool_resources, "file_search", None)
        vector_store_ids = (
            file_search_resource.vector_store_ids
            if (file_search_resource and hasattr(file_search_resource, "vector_store_ids"))
            else []
        )

        # Ensure vector store exists
        if vector_store_ids:
            vector_store_id = vector_store_ids[0]
        else:
            # No vector store associated yet, create one
            logging.info("No associated vector store found. Creating a new one.")
            vector_store = client.beta.vector_stores.create(name=f"Assistant_{assistant}_Store")
            vector_store_id = vector_store.id

            # Ensure the 'file_search' tool is present in the assistant's tools
            existing_tools = assistant_obj.tools if assistant_obj.tools else []
            if not any(t["type"] == "file_search" for t in existing_tools):
                existing_tools.append({"type": "file_search"})
                
            # Update assistant with vector store
            client.beta.assistants.update(
                assistant_id=assistant,
                tools=existing_tools,
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [vector_store_id]
                    }
                }
            )

        # Check for code_interpreter tool and add if missing
        existing_tools = assistant_obj.tools if assistant_obj.tools else []
        if not any(t["type"] == "code_interpreter" for t in existing_tools):
            existing_tools.append({"type": "code_interpreter"})
            client.beta.assistants.update(
                assistant_id=assistant,
                tools=existing_tools
            )

        # Get current code_interpreter file_ids if any
        code_interpreter_resource = getattr(assistant_obj.tool_resources, "code_interpreter", None)
        existing_file_ids = (
            code_interpreter_resource.file_ids
            if (code_interpreter_resource and hasattr(code_interpreter_resource, "file_ids"))
            else []
        )

        # Handle file based on type
        file_id = None
        file_type = None
        
        if is_csv or is_excel:
            # Upload to assistants files API for code_interpreter
            with open(file_path, "rb") as file_stream:
                uploaded_file = client.files.create(
                    file=file_stream,
                    purpose="assistants"
                )
                file_id = uploaded_file.id
                file_type = "csv" if is_csv else "excel"
                
                # Add new file_id to existing ones
                updated_file_ids = existing_file_ids + [file_id]
                
                # Update assistant tool resources
                tool_resources = assistant_obj.tool_resources if hasattr(assistant_obj, "tool_resources") else {}
                
                # Ensure both file_search and code_interpreter are present
                updated_tool_resources = {
                    "file_search": {"vector_store_ids": [vector_store_id]},
                    "code_interpreter": {"file_ids": updated_file_ids}
                }
                
                client.beta.assistants.update(
                    assistant_id=assistant,
                    tool_resources=updated_tool_resources
                )
                logging.info(f"CSV/Excel file uploaded and attached to code_interpreter: {file_id}")
                
        elif is_img:
            # For images, perform analysis
            analysis_text = await image_analysis(client, file_content, file.filename, prompt)
            
            # Save analysis to file
            analysis_file_path = os.path.join(FILE_STORAGE_DIR, f"{file.filename}_analysis.txt")
            with open(analysis_file_path, 'w') as f:
                f.write(analysis_text)
                
            file_type = "image"
            
            # Add to vector store for text search on the analysis
            with open(analysis_file_path, "rb") as analysis_file:
                client.beta.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=[analysis_file]
                )
                
            # Add image analysis to thread if thread_id provided
            if thread_id:
                client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=f"Image Analysis for {file.filename}: {analysis_text}"
                )
                logging.info(f"Added image analysis to thread {thread_id}")
        else:
            # For other document types, add to vector store
            with open(file_path, "rb") as file_stream:
                file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=[file_stream]
                )
                logging.info(f"Document file uploaded to vector store: {file.filename}")
                file_type = "document"

        # Add file metadata to thread if thread exists
        if thread_id:
            file_info = {
                "name": file.filename,
                "type": file_type,
                "path": file_path,
                "file_id": file_id,
                "vector_store_id": vector_store_id
            }
            
            if is_img:
                file_info["analysis"] = analysis_text
                
            await add_file_metadata_to_thread(client, thread_id, file_info)
            
        # If context provided and thread exists, update context
        if context and thread_id:
            try:
                await update_context(client, thread_id, context)
            except Exception as e:
                logging.error(f"Error updating context in thread: {e}")
                # Continue even if context update fails

        return JSONResponse(
            {
                "message": "File successfully uploaded and processed.",
                "file_type": file_type,
                "file_id": file_id,
                "image_analyzed": is_img
            },
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/conversation")
async def conversation(
    session: Optional[str] = None,
    prompt: Optional[str] = None,
    assistant: Optional[str] = None,
    context: Optional[str] = None,  # New: Optional context parameter
    **kwargs
):
    """
    Handles conversation queries. 
    Preserves the original query parameters and output format.
    Additional parameters are accepted but ignored.
    """
    client = create_client()

    try:
        # If no assistant or session provided, create them (same fallback approach)
        if not assistant:
            assistant_obj = client.beta.assistants.create(
                name="conversation_assistant",
                model="gpt-4o-mini",
                instructions="You are a conversation assistant."
            )
            assistant = assistant_obj.id

        if not session:
            thread = client.beta.threads.create()
            session = thread.id
            
        # If context is provided, update user persona context
        if context:
            await update_context(client, session, context)

        # Add message if prompt given
        if prompt:
            client.beta.threads.messages.create(
                thread_id=session,
                role="user",
                content=prompt
            )

        def stream_response():
            buffer = []
            try:
                with client.beta.threads.runs.stream(thread_id=session, assistant_id=assistant) as stream:
                    for text in stream.text_deltas:
                        buffer.append(text)
                        if len(buffer) >= 10:
                            yield ''.join(buffer)
                            buffer = []
                if buffer:
                    yield ''.join(buffer)
            except Exception as e:
                logging.error(f"Streaming error: {e}")
                yield "[ERROR] The response was interrupted. Please try again."

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to process conversation")


@app.get("/chat")
async def chat(
    session: Optional[str] = None,
    prompt: Optional[str] = None,
    assistant: Optional[str] = None,
    context: Optional[str] = None,  # New: Optional context parameter
    **kwargs
):
    """
    Handles conversation queries.
    Preserves the original query parameters and output format.
    Additional parameters are accepted but ignored.
    """
    client = create_client()

    try:
        # If no assistant or session provided, create them
        if not assistant:
            assistant_obj = client.beta.assistants.create(
                name="chat_assistant",
                model="gpt-4o-mini",
                instructions="You are a conversation assistant."
            )
            assistant = assistant_obj.id

        if not session:
            thread = client.beta.threads.create()
            session = thread.id
            
        # If context is provided, update user persona context
        if context:
            await update_context(client, session, context)

        # Add message if prompt given
        if prompt:
            client.beta.threads.messages.create(
                thread_id=session,
                role="user",
                content=prompt
            )

        response_text = []
        try:
            with client.beta.threads.runs.stream(thread_id=session, assistant_id=assistant) as stream:
                for text in stream.text_deltas:
                    response_text.append(text)
        except Exception as e:
            logging.error(f"Streaming error: {e}")
            raise HTTPException(status_code=500, detail="The response was interrupted. Please try again.")

        full_response = ''.join(response_text)
        return JSONResponse(content={"response": full_response})

    except Exception as e:
        logging.error(f"Error in conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to process conversation")


@app.post("/trim-thread")
async def trim_thread(request: Request, assistant_id: str = None, max_age_days: Optional[int] = None, **kwargs):
    """
    Gets all threads for a given assistant, summarizes them, and removes old threads.
    Uses 48 hours as the threshold for thread cleanup.
    Accepts both query parameters and form data.
    """
    # Get parameters from form data if not provided in query
    if not assistant_id:
        form_data = await request.form()
        assistant_id = form_data.get("assistant_id")
    
    # Set default cleanup threshold to 48 hours
    time_threshold_hours = 48
    
    if not assistant_id:
        raise HTTPException(status_code=400, detail="assistant_id is required")
    
    client = create_client()
    summary_store = {}
    deleted_count = 0
    summarized_count = 0
    
    try:
        # Step 1: Get all runs to identify threads used with this assistant
        all_threads = {}
        
        # Get all runs (limited by API, may need pagination in production)
        runs = client.beta.threads.runs.list_all_runs()
        
        # Filter runs by assistant_id and collect their thread_ids
        for run in runs.data:
            if run.assistant_id == assistant_id:
                thread_id = run.thread_id
                # Get the last_active timestamp (using the run's created_at as proxy)
                last_active = datetime.datetime.fromtimestamp(run.created_at)
                
                if thread_id in all_threads:
                    # Keep the most recent timestamp
                    if last_active > all_threads[thread_id]['last_active']:
                        all_threads[thread_id]['last_active'] = last_active
                else:
                    all_threads[thread_id] = {
                        'thread_id': thread_id,
                        'last_active': last_active
                    }
        
        # Sort threads by last_active time (most recent first)
        sorted_threads = sorted(
            all_threads.values(), 
            key=lambda x: x['last_active'], 
            reverse=True
        )
        
        # Get current time for age comparison
        now = datetime.datetime.now()
        
        # Step 2: Process each thread
        for thread_info in sorted_threads:
            thread_id = thread_info['thread_id']
            last_active = thread_info['last_active']
            
            # Calculate hours since last activity
            thread_age_hours = (now - last_active).total_seconds() / 3600
            
            # Skip active threads that are recent
            if thread_age_hours <= 1:  # Keep very recent threads untouched
                continue
                
            # Check if thread has summary metadata
            try:
                thread = client.beta.threads.retrieve(thread_id=thread_id)
                metadata = thread.metadata if hasattr(thread, 'metadata') else {}
                
                # If it's a summary thread and too old, delete it
                if metadata.get('is_summary') and thread_age_hours > time_threshold_hours:
                    client.beta.threads.delete(thread_id=thread_id)
                    deleted_count += 1
                    continue
                
                # If regular thread and older than threshold, summarize it
                if thread_age_hours > time_threshold_hours:
                    # Get messages in the thread
                    messages = client.beta.threads.messages.list(thread_id=thread_id)
                    
                    if len(list(messages.data)) > 0:
                        # Create prompt for summarization
                        summary_content = "\n\n".join([
                            f"{msg.role}: {msg.content[0].text.value if hasattr(msg, 'content') and len(msg.content) > 0 else 'No content'}" 
                            for msg in messages.data
                        ])
                        
                        # Create a new thread for the summary
                        summary_thread = client.beta.threads.create(
                            metadata={"is_summary": True, "original_thread_id": thread_id}
                        )
                        
                        # Add a request to summarize
                        client.beta.threads.messages.create(
                            thread_id=summary_thread.id,
                            role="user",
                            content=f"Summarize the following conversation in a concise paragraph:\n\n{summary_content}"
                        )
                        
                        # Run the summarization
                        run = client.beta.threads.runs.create(
                            thread_id=summary_thread.id,
                            assistant_id=assistant_id
                        )
                        
                        # Wait for completion with timeout
                        max_wait = 30  # 30 seconds timeout
                        start_time = time.time()
                        
                        while True:
                            if time.time() - start_time > max_wait:
                                logging.warning(f"Timeout waiting for summarization of thread {thread_id}")
                                break
                                
                            run_status = client.beta.threads.runs.retrieve(
                                thread_id=summary_thread.id,
                                run_id=run.id
                            )
                            
                            if run_status.status == "completed":
                                # Get the summary
                                summary_messages = client.beta.threads.messages.list(
                                    thread_id=summary_thread.id,
                                    order="desc"
                                )
                                
                                # Extract the summary text
                                summary_text = next(
                                    (msg.content[0].text.value for msg in summary_messages.data 
                                     if msg.role == "assistant" and hasattr(msg, 'content') and len(msg.content) > 0),
                                    "Summary not available."
                                )
                                
                                # Store summary info
                                summary_store[thread_id] = {
                                    "summary": summary_text,
                                    "summary_thread_id": summary_thread.id,
                                    "original_thread_id": thread_id,
                                    "summarized_at": now.isoformat()
                                }
                                
                                # Delete the original thread
                                client.beta.threads.delete(thread_id=thread_id)
                                deleted_count += 1
                                summarized_count += 1
                                break
                            
                            elif run_status.status in ["failed", "cancelled", "expired"]:
                                logging.error(f"Summary generation failed for thread {thread_id}: {run_status.status}")
                                break
                                
                            time.sleep(1)
            
            except Exception as e:
                logging.error(f"Error processing thread {thread_id}: {e}")
                continue
        
        return JSONResponse({
            "status": "Thread trimming completed",
            "threads_processed": len(sorted_threads),
            "threads_summarized": summarized_count,
            "threads_deleted": deleted_count,
            "summaries_stored": len(summary_store)
        })
        
    except Exception as e:
        logging.error(f"Error in trim-thread: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trim threads: {str(e)}")


@app.post("/file-cleanup")
async def file_cleanup(request: Request, assistant_id: str = None, **kwargs):
    """
    Cleans up files associated with an assistant, including code_interpreter files.
    Removes files that are older than 48 hours.
    """
    # Get parameters from form data if not provided in query
    if not assistant_id:
        form_data = await request.form()
        assistant_id = form_data.get("assistant_id")
    
    if not assistant_id:
        raise HTTPException(status_code=400, detail="assistant_id is required")
    
    client = create_client()
    vector_store_files_deleted = 0
    code_interpreter_files_deleted = 0
    
    try:
        # Retrieve the assistant to get tool resources
        assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
        
        # Get current time for age comparison
        now = datetime.datetime.now()
        
        # Step 1: Clean up vector store files
        if hasattr(assistant, "tool_resources") and hasattr(assistant.tool_resources, "file_search"):
            vector_store_ids = assistant.tool_resources.file_search.vector_store_ids
            
            for vector_store_id in vector_store_ids:
                # Get all file batches in the vector store
                file_batches = client.beta.vector_stores.file_batches.list(vector_store_id=vector_store_id)
                
                # Process each file batch to find files older than 48 hours
                for batch in file_batches.data:
                    # Calculate age in hours
                    batch_created = datetime.datetime.fromtimestamp(batch.created_at)
                    batch_age_hours = (now - batch_created).total_seconds() / 3600
                    
                    # Skip recent batches
                    if batch_age_hours <= 48:
                        continue
                        
                    # Get files in this batch
                    files = client.beta.vector_stores.files.list(
                        vector_store_id=vector_store_id,
                        file_batch_id=batch.id
                    )
                    
                    # Delete files older than 48 hours
                    for file in files.data:
                        try:
                            client.beta.vector_stores.files.delete(
                                vector_store_id=vector_store_id,
                                file_id=file.id
                            )
                            vector_store_files_deleted += 1
                        except Exception as e:
                            logging.error(f"Error deleting vector store file {file.id}: {e}")
        
        # Step 2: Clean up code_interpreter files
        if hasattr(assistant, "tool_resources") and hasattr(assistant.tool_resources, "code_interpreter"):
            code_interpreter_file_ids = assistant.tool_resources.code_interpreter.file_ids
            
            # Get files info to check their age
            files_to_keep = []
            
            for file_id in code_interpreter_file_ids:
                try:
                    file_info = client.files.retrieve(file_id=file_id)
                    file_created = datetime.datetime.fromtimestamp(file_info.created_at)
                    file_age_hours = (now - file_created).total_seconds() / 3600
                    
                    # Keep files newer than 48 hours
                    if file_age_hours <= 48:
                        files_to_keep.append(file_id)
                    else:
                        # Delete older files
                        client.files.delete(file_id=file_id)
                        code_interpreter_files_deleted += 1
                except Exception as e:
                    logging.error(f"Error processing code_interpreter file {file_id}: {e}")
                    # Keep the file if we can't determine its age
                    files_to_keep.append(file_id)
            
            # Update assistant with remaining files
            if len(code_interpreter_file_ids) != len(files_to_keep):
                # Create updated tool_resources
                updated_tool_resources = assistant.tool_resources
                updated_tool_resources.code_interpreter.file_ids = files_to_keep
                
                # Update the assistant
                client.beta.assistants.update(
                    assistant_id=assistant_id,
                    tool_resources=updated_tool_resources
                )
        
        return JSONResponse({
            "status": "File cleanup completed",
            "assistant_id": assistant_id,
            "vector_store_files_deleted": vector_store_files_deleted,
            "code_interpreter_files_deleted": code_interpreter_files_deleted
        })
        
    except Exception as e:
        logging.error(f"Error in file-cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clean up files: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
