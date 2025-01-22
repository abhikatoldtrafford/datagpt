import base64
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import AzureOpenAI
from typing import List, Optional
import time

app = FastAPI()

AZURE_ENDPOINT = "https://kb-stellar.openai.azure.com/"
AZURE_API_VERSION = "2024-08-01-preview"
AZURE_API_KEY = "bc0ba854d3644d7998a5034af62d03ce"


def create_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

@app.post("/data-analysis")
async def data_analysis(
    files: List[UploadFile] = File(...),
    user_input: str = Form("Analyze the provided data and generate insights"),
    product_id: Optional[str] = None
):
    """
    Endpoint for data analysis using Azure OpenAI's Code Interpreter.
    Accepts multiple files and a user query, returns analysis results with images.
    """
    client = create_client()
    assistant = None
    thread = None

    try:
        # 1. Upload files to Azure OpenAI
        file_ids = []
        for file in files:
            try:
                file_content = await file.read()
                uploaded_file = client.files.create(
                    file=file_content,
                    purpose="assistants"
                )
                file_ids.append(uploaded_file.id)
                await file.seek(0)  # Reset file pointer for potential reuse
            except Exception as e:
                logging.error(f"Error uploading {file.filename}: {e}")
                raise HTTPException(400, f"Failed to upload {file.filename}")

        # 2. Create assistant with code interpreter and files
        assistant = client.beta.assistants.create(
            name="Data Analyst Assistant",
            instructions=(
                "You are a senior data analyst. Use Python (pandas, matplotlib, etc) to analyze files. "
                "For numerical data, always show statistical analysis. For images, provide insights. "
                "When appropriate, create visualizations to support your findings."
            ),
            model="gpt-4o-mini",
            tools=[{"type": "code_interpreter"}],
            tool_resources={"code_interpreter": {"file_ids": file_ids}}
        )

        # 3. Create thread and add user message
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input,
            file_ids=file_ids
        )

        # 4. Run analysis
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Provide detailed analysis with visualizations when helpful."
        )

        # 5. Wait for completion
        start_time = time.time()
        while True:
            if time.time() - start_time > 120:  # 2 minute timeout
                raise HTTPException(504, "Analysis timed out")

            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise HTTPException(500, f"Analysis failed: {run_status.last_error}")
            time.sleep(2)

        # 6. Retrieve and format response
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
                        try:
                            # Get image bytes
                            image_data = client.files.content(content.image_file.file_id)
                            image_bytes = image_data.read()
                            
                            # Convert to base64
                            base64_image = base64.b64encode(image_bytes).decode("utf-8")
                            
                            response_content.append({
                                "type": "image",
                                "format": "png",  # Azure currently only returns PNG
                                "content": base64_image
                            })
                        except Exception as e:
                            logging.error(f"Error processing image: {e}")
                            continue

        return JSONResponse({
            "analysis": response_content,
            "assistant_id": assistant.id,
            "thread_id": thread.id
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        # Cleanup resources
        try:
            if thread:
                client.beta.threads.delete(thread.id)
            if assistant:
                client.beta.assistants.delete(assistant.id)
        except Exception as e:
            logging.warning(f"Cleanup error: {str(e)}")
