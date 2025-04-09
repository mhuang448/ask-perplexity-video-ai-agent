# app/pipeline_logic.py
import time
import json
from datetime import datetime, timezone
from botocore.exceptions import ClientError

# Import helper functions and clients from utils
from .utils import S3_CLIENT, CONFIG, get_s3_json_path, determine_if_preprocessed
from .models import VideoMetadata, Interaction # Import Pydantic models for structure

# Placeholder imports for AI clients - replace with actual imports
# import pinecone
# import openai

# --- S3 JSON Read/Write Helpers (Crucial for State) ---

def get_processing_status_from_s3(bucket: str, key: str) -> dict:
    """Reads the JSON metadata file from S3 and returns it as a dict."""
    try:
        response = S3_CLIENT.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        print(f"Successfully read status from s3://{bucket}/{key}")
        return data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"S3 Key not found: s3://{bucket}/{key}")
            raise FileNotFoundError(f"Metadata file not found at {key}")
        else:
            print(f"Error reading from S3 s3://{bucket}/{key}: {e}")
            raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from s3://{bucket}/{key}: {e}")
        raise ValueError("Invalid JSON content in S3 file")
    except Exception as e:
        print(f"Unexpected error reading status from s3://{bucket}/{key}: {e}")
        raise


def update_interaction_status_in_s3(bucket: str, key: str, interaction_id: str, status: str, final_answer: str = None, answer_timestamp: str = None):
    """Reads the S3 JSON, updates a specific interaction, writes it back."""
    print(f"Attempting to update interaction {interaction_id} in s3://{bucket}/{key} to status: {status}")
    retries = 3
    for attempt in range(retries):
        try:
            # 1. GET current JSON
            response = S3_CLIENT.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)
            # Use Pydantic model for validation (optional but good)
            # metadata = VideoMetadata(**data)

            # 2. Find and Update Interaction
            interaction_found = False
            if 'interactions' not in data:
                data['interactions'] = []

            for interaction in data['interactions']:
                if interaction.get('interaction_id') == interaction_id:
                    interaction['status'] = status
                    if final_answer is not None:
                        interaction['ai_answer'] = final_answer
                        interaction['answer_timestamp'] = answer_timestamp or datetime.now(timezone.utc).isoformat()
                    interaction_found = True
                    break

            if not interaction_found:
                print(f"Warning: Interaction ID {interaction_id} not found in {key}. Cannot update status.")
                # Decide if you should add it here or if it should always exist first
                # For now, we'll just log and proceed if not found

            # 3. PUT updated JSON back
            S3_CLIENT.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(data, indent=2),
                ContentType='application/json'
            )
            print(f"Successfully updated interaction {interaction_id} status to {status} in s3://{bucket}/{key}")
            return # Success, exit retry loop

        except ClientError as e:
            # Handle potential concurrency issues (ConditionalCheckFailedException if using versioning/ETags)
            # Handle throttling
            print(f"S3 ClientError on attempt {attempt + 1} updating {key}: {e}")
            if attempt == retries - 1: raise # Raise after last attempt
            time.sleep(2 ** attempt) # Exponential backoff

        except Exception as e:
            print(f"Error updating interaction status in {key} on attempt {attempt + 1}: {e}")
            if attempt == retries - 1: raise # Raise after last attempt
            time.sleep(2 ** attempt) # Exponential backoff

def update_overall_processing_status(bucket: str, key: str, overall_status: str):
     """Reads the S3 JSON, updates the top-level processing_status, writes it back."""
     # Similar Read-Modify-Write logic as update_interaction_status_in_s3
     # ... (Implementation needed, handle retries) ...
     print(f"Updating overall status for {key} to {overall_status}")
     # TODO: Implement this function robustly

# --- Placeholder Functions for Pipeline Steps ---

def _download_video(video_url: str, local_path: str):
    print(f"Placeholder: Downloading {video_url} to {local_path}...")
    time.sleep(2) # Simulate download time
    # TODO: Implement actual download using yt-dlp or requests
    print("Placeholder: Download complete.")
    # Return path to downloaded file
    return local_path

def _upload_to_s3(local_path: str, bucket: str, s3_key: str):
    print(f"Placeholder: Uploading {local_path} to s3://{bucket}/{s3_key}...")
    # TODO: Implement S3 upload using S3_CLIENT.upload_file
    time.sleep(1)
    print("Placeholder: Upload complete.")

def _chunk_video(video_s3_key: str, bucket: str) -> list:
    print(f"Placeholder: Chunking video s3://{bucket}/{video_s3_key}...")
    time.sleep(5) # Simulate chunking time
    # TODO: Implement chunking
    #   - Download video from S3 (or stream)
    #   - Use PySceneDetect or ffmpeg
    #   - Upload chunks to S3 (e.g., under /chunks/ prefix)
    #   - Generate chunk metadata (start/end times, names, etc.)
    print("Placeholder: Chunking complete.")
    # Return list of chunk metadata dictionaries
    return [
        {"chunk_name": "chunk1.mp4", "start_timestamp": "00:00.000", "end_timestamp": "00:04.000", "metadata_1": "metadata_1_content", "metadata_2": "metadata_2_content"},
        {"chunk_name": "chunk2.mp4", "start_timestamp": "00:04.000", "end_timestamp": "00:08.000", "metadata_1": "metadata_1_content", "metadata_2": "metadata_2_content"}
    ] # Example

def _generate_captions_and_summary(chunks_metadata: list, video_s3_base_path: str, bucket: str) -> tuple[list, str, list]:
     print("Placeholder: Generating captions and summary...")
     time.sleep(10) # Simulate AI calls
     # TODO: Implement caption/summary generation
     #   - For each chunk:
     #     - Download chunk from S3 (or use presigned URL)
     #     - Call Gemini API to generate caption
     #     - Update the corresponding chunk metadata dict with the caption
     #   - Concatenate all captions
     #   - Call OpenAI API to generate overall summary and key themes
     updated_chunks_metadata = chunks_metadata # Add captions to this list
     overall_summary = "This is a placeholder summary."
     key_themes = ["placeholder", "example"]
     print("Placeholder: Captioning/Summarization complete.")
     return updated_chunks_metadata, overall_summary, key_themes

def _index_captions(video_id: str, chunks_with_captions: list):
    print(f"Placeholder: Indexing captions for {video_id}...")
    time.sleep(3) # Simulate embedding/indexing
    # TODO: Implement indexing
    #   - Initialize Pinecone client
    #   - For each chunk with a caption:
    #     - Get caption text
    #     - Call OpenAI Embedding API
    #     - Prepare vector object (id=chunk_name, values=embedding, metadata={video_id, caption_text, start, end,...})
    #     - Upsert vectors to Pinecone in batches
    print("Placeholder: Indexing complete.")

def _retrieve_relevant_chunks(video_id: str, user_query: str) -> list:
    print(f"Placeholder: Retrieving relevant chunks for query: '{user_query}'...")
    time.sleep(1) # Simulate embedding query + Pinecone search
    # TODO: Implement retrieval
    #   - Initialize Pinecone client
    #   - Get embedding for user_query (OpenAI)
    #   - Query Pinecone index, filtering by video_id metadata field
    #   - Return top_k matching results (including metadata)
    print("Placeholder: Retrieval complete.")
    # Return list of retrieved chunk metadata dictionaries from Pinecone results
    return [{"caption": "Placeholder retrieved caption 1", "start_timestamp": "00:01.000", "metadata_3": "metadata_3_content"}] # Example

def _assemble_context(retrieved_chunks: list, summary: str, themes: list) -> str:
    print("Placeholder: Assembling context...")
    # TODO: Implement context assembly based on README description
    context = f"Summary: {summary}\\nThemes: {', '.join(themes)}\\n\\nRelevant Clips:\\n"
    # Add details from retrieved_chunks
    print("Placeholder: Context assembly complete.")
    return context # Return formatted context string

def _call_mcp(context: str, user_query: str):
    print("Placeholder: Calling Perplexity MCP...")
    time.sleep(4) # Simulate MCP call
    # TODO: Implement MCP client interaction
    #   - Choose tool (rule-based or Claude-orchestrated)
    #   - Call session.call_tool
    mcp_result = "This is a placeholder result from MCP."
    print("Placeholder: MCP call complete.")
    return mcp_result

def _synthesize_answer(context: str, mcp_result: str, user_query: str) -> str:
    print("Placeholder: Synthesizing final answer...")
    time.sleep(3) # Simulate final LLM call
    # TODO: Implement final answer synthesis
    #   - Construct prompt for OpenAI completion model (gpt-4o-mini)
    #   - Include original query, video context, MCP result
    #   - Call OpenAI API
    final_answer = f"Placeholder answer for '{user_query}' based on context and MCP."
    print("Placeholder: Synthesis complete.")
    return final_answer


# --- Background Task Implementations ---

def run_query_pipeline_async(video_id: str, user_query: str, interaction_id: str, query_timestamp: str, s3_json_path: str, s3_bucket: str):
    """Background task to answer a query for a PRE-PROCESSED video."""
    print(f"BACKGROUND TASK: Starting query pipeline for interaction {interaction_id} on video {video_id}")
    start_time = time.time()

    try:
        # 1. Mark as processing (ensure interaction exists first)
        # It's better practice to add the interaction shell BEFORE starting the task,
        # but for PoC this update confirms the task has started trying.
        update_interaction_status_in_s3(s3_bucket, s3_json_path, interaction_id, "processing")

        # 2. Load metadata (needed for context assembly)
        metadata = get_processing_status_from_s3(s3_bucket, s3_json_path)
        summary = metadata.get("overall_summary", "Summary not found.")
        themes = metadata.get("key_themes", [])

        # 3. Retrieve relevant chunks from Pinecone
        retrieved_chunks = _retrieve_relevant_chunks(video_id, user_query)

        # 4. Assemble context
        context = _assemble_context(retrieved_chunks, summary, themes)

        # 5. Call MCP tool
        mcp_result = _call_mcp(context, user_query)

        # 6. Synthesize final answer
        final_answer = _synthesize_answer(context, mcp_result, user_query)

        # 7. Update status to completed with the answer
        update_interaction_status_in_s3(
            s3_bucket, s3_json_path, interaction_id, "completed",
            final_answer=final_answer,
            answer_timestamp=datetime.now(timezone.utc).isoformat()
        )
        print(f"BACKGROUND TASK: Query pipeline for interaction {interaction_id} COMPLETED.")

    except Exception as e:
        print(f"BACKGROUND TASK ERROR: Query pipeline for interaction {interaction_id} FAILED: {e}")
        try:
            # Attempt to mark as failed
            update_interaction_status_in_s3(s3_bucket, s3_json_path, interaction_id, "failed")
        except Exception as update_e:
            print(f"BACKGROUND TASK ERROR: Failed to update status to failed for {interaction_id}: {update_e}")
    finally:
        end_time = time.time()
        print(f"BACKGROUND TASK: Query pipeline for interaction {interaction_id} finished in {end_time - start_time:.2f} seconds.")


def run_full_pipeline_async(video_url: str, user_query: str, video_id: str, s3_video_base_path: str, s3_json_path: str, s3_bucket: str, interaction_id: str, query_timestamp: str):
    """Background task for the FULL pipeline: Download -> ... -> Answer."""
    print(f"BACKGROUND TASK: Starting FULL pipeline for video {video_id} (interaction {interaction_id})")
    start_time = time.time()
    current_overall_status = "processing_download" # Initial status

    try:
        # 1. Create Initial JSON & Mark initial status
        initial_metadata = {
            "video_id": video_id,
            "source_url": video_url,
            "processing_status": current_overall_status,
            "interactions": [
                {
                    "interaction_id": interaction_id,
                    "user_query": user_query,
                    "query_timestamp": query_timestamp,
                    "status": "pending", # Mark Q&A as pending until video is processed
                    "ai_answer": None,
                    "answer_timestamp": None
                }
            ],
            # Initialize other fields as empty/null
             "overall_summary": None, "key_themes": [], "total_duration_seconds": None, "chunks": []
        }
        S3_CLIENT.put_object(
            Bucket=s3_bucket, Key=s3_json_path, Body=json.dumps(initial_metadata, indent=2), ContentType='application/json'
        )
        print(f"Initial metadata saved to {s3_json_path}")

        # 2. Download Video
        # TODO: Define local download path (maybe temp dir?)
        local_video_path = f"/tmp/{video_id}.mp4" # Example path in container
        _download_video(video_url, local_video_path)
        s3_video_key = f"{s3_video_base_path}.mp4"
        _upload_to_s3(local_video_path, s3_bucket, s3_video_key)
        current_overall_status = "processing_chunking"
        update_overall_processing_status(s3_bucket, s3_json_path, current_overall_status)
        # TODO: Clean up local downloaded file os.remove(local_video_path)

        # 3. Chunk Video
        chunks_metadata_no_captions = _chunk_video(s3_video_key, s3_bucket)
        # Update JSON with chunk info (no captions yet)
        # ... (read, update 'chunks' field, write back) ...
        current_overall_status = "processing_captioning"
        update_overall_processing_status(s3_bucket, s3_json_path, current_overall_status)

        # 4. Generate Captions & Summary
        chunks_with_captions, summary, themes = _generate_captions_and_summary(chunks_metadata_no_captions, s3_video_base_path, s3_bucket)
        # Update JSON with captions, summary, themes
        # ... (read, update 'chunks', 'overall_summary', 'key_themes', write back) ...
        current_overall_status = "processing_indexing"
        update_overall_processing_status(s3_bucket, s3_json_path, current_overall_status)

        # 5. Index Captions
        _index_captions(video_id, chunks_with_captions)
        current_overall_status = "completed" # Video processing done!
        update_overall_processing_status(s3_bucket, s3_json_path, current_overall_status)
        # TODO: Clean up temporary S3 chunks if desired

        # 6. Now, process the initial query using the generated data
        print(f"Video processing complete for {video_id}. Now running query part for interaction {interaction_id}")
        update_interaction_status_in_s3(s3_bucket, s3_json_path, interaction_id, "processing") # Mark Q&A as processing

        retrieved_chunks = _retrieve_relevant_chunks(video_id, user_query)
        context = _assemble_context(retrieved_chunks, summary, themes)
        mcp_result = _call_mcp(context, user_query)
        final_answer = _synthesize_answer(context, mcp_result, user_query)

        update_interaction_status_in_s3(
            s3_bucket, s3_json_path, interaction_id, "completed",
            final_answer=final_answer,
            answer_timestamp=datetime.now(timezone.utc).isoformat()
        )
        print(f"BACKGROUND TASK: Full pipeline for interaction {interaction_id} COMPLETED.")

    except Exception as e:
        print(f"BACKGROUND TASK ERROR: Full pipeline for video {video_id} / interaction {interaction_id} FAILED at step '{current_overall_status}': {e}")
        try:
            # Attempt to mark overall video and specific interaction as failed
            update_overall_processing_status(s3_bucket, s3_json_path, "failed")
            update_interaction_status_in_s3(s3_bucket, s3_json_path, interaction_id, "failed")
        except Exception as update_e:
            print(f"BACKGROUND TASK ERROR: Failed to update status to failed for {interaction_id} after error: {update_e}")
    finally:
        end_time = time.time()
        print(f"BACKGROUND TASK: Full pipeline for interaction {interaction_id} finished in {end_time - start_time:.2f} seconds.")

