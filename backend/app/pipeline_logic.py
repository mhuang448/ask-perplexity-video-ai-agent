# app/pipeline_logic.py
import time
import json
import asyncio # Added for MCP async operations
from contextlib import AsyncExitStack # Added for MCP connection management
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Tuple, Optional # Added List, Dict, Any, Tuple, Optional

# Import helper functions and clients from utils
from .utils import (
    S3_CLIENT, CONFIG, get_s3_json_path, get_s3_interactions_path,
    OPENAI_CLIENT, PINECONE_INDEX,
    ANTHROPIC_CLIENT, # Added Anthropic client (optional)
    get_mcp_server_params # Added MCP helper
)
from .models import VideoMetadata, Interaction # Import Pydantic models for structure

# Import specific exceptions for better handling
from openai import OpenAIError
from pinecone.exceptions import PineconeException
# Import MCP components
from mcp import ClientSession
from mcp.client.stdio import stdio_client
# Import Anthropic specific exceptions
from anthropic import APIError as AnthropicAPIError

# Placeholder imports for AI clients - replace with actual imports
# import pinecone
# import openai

# --- S3 JSON Read/Write Helpers (Crucial for State) ---

def get_video_metadata_from_s3(bucket: str, key: str) -> dict:
    """Reads the JSON metadata file from S3 and returns it as a dict."""
    try:
        response = S3_CLIENT.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        print(f"Successfully read metadata from s3://{bucket}/{key}")
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
        print(f"Unexpected error reading metadata from s3://{bucket}/{key}: {e}")
        raise

def get_interactions_from_s3(bucket: str, key: str) -> list:
    """Reads the interactions.json file from S3 and returns the list of interactions.
    If the file doesn't exist, returns an empty list."""
    try:
        response = S3_CLIENT.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        interactions = data.get('interactions', [])
        print(f"Successfully read {len(interactions)} interactions from s3://{bucket}/{key}")
        return interactions
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"Interactions file not found: s3://{bucket}/{key} - This is normal for first query")
            return []  # Return empty list for first interaction
        else:
            print(f"Error reading interactions from S3 s3://{bucket}/{key}: {e}")
            raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from s3://{bucket}/{key}: {e}")
        raise ValueError("Invalid JSON content in interactions file")
    except Exception as e:
        print(f"Unexpected error reading interactions from s3://{bucket}/{key}: {e}")
        raise

def add_interaction_to_s3(bucket: str, key: str, interaction: dict):
    """Adds a new interaction to the interactions.json file.
    Creates the file if it doesn't exist."""
    print(f"Adding interaction {interaction.get('interaction_id')} to s3://{bucket}/{key}")
    retries = 3
    
    for attempt in range(retries):
        try:
            # Get existing interactions or start with empty list
            try:
                interactions = get_interactions_from_s3(bucket, key)
            except:
                interactions = []
            
            # Add the new interaction
            interactions.append(interaction)
            
            # Write back to S3
            data = {'interactions': interactions}
            S3_CLIENT.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(data, indent=2),
                ContentType='application/json'
            )
            print(f"Successfully added interaction {interaction.get('interaction_id')} to s3://{bucket}/{key}")
            return  # Success, exit retry loop
            
        except ClientError as e:
            print(f"S3 ClientError on attempt {attempt + 1} adding interaction to {key}: {e}")
            if attempt == retries - 1: 
                raise  # Raise after last attempt
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            print(f"Error adding interaction to {key} on attempt {attempt + 1}: {e}")
            if attempt == retries - 1: 
                raise  # Raise after last attempt
            time.sleep(2 ** attempt)  # Exponential backoff

def update_interaction_status_in_s3(bucket: str, key: str, interaction_id: str, status: str, final_answer: str = None, answer_timestamp: str = None):
    """Updates an existing interaction's status in the interactions.json file."""
    print(f"Attempting to update interaction {interaction_id} in s3://{bucket}/{key} to status: {status}")
    retries = 3
    for attempt in range(retries):
        try:
            # 1. GET current interactions
            interactions = get_interactions_from_s3(bucket, key)
            
            # 2. Find and Update Interaction
            interaction_found = False
            for interaction in interactions:
                if interaction.get('interaction_id') == interaction_id:
                    interaction['status'] = status
                    if final_answer is not None:
                        interaction['ai_answer'] = final_answer
                        interaction['answer_timestamp'] = answer_timestamp or datetime.now(timezone.utc).isoformat()
                    interaction_found = True
                    break

            if not interaction_found:
                print(f"Warning: Interaction ID {interaction_id} not found in {key}. Cannot update status.")
                # Could add logic to create it if missing, but this shouldn't happen in normal flow
                return

            # 3. PUT updated interactions back
            data = {'interactions': interactions}
            S3_CLIENT.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(data, indent=2),
                ContentType='application/json'
            )
            print(f"Successfully updated interaction {interaction_id} status to {status} in s3://{bucket}/{key}")
            return # Success, exit retry loop

        except ClientError as e:
            print(f"S3 ClientError on attempt {attempt + 1} updating {key}: {e}")
            if attempt == retries - 1: 
                raise  # Raise after last attempt
            time.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            print(f"Error updating interaction status in {key} on attempt {attempt + 1}: {e}")
            if attempt == retries - 1: 
                raise  # Raise after last attempt
            time.sleep(2 ** attempt)  # Exponential backoff

def update_overall_processing_status(bucket: str, key: str, overall_status: str):
    """Reads the S3 JSON, updates the top-level processing_status, writes it back."""
    print(f"Updating overall status for {key} to {overall_status}")
    retries = 3
    
    for attempt in range(retries):
        try:
            # 1. GET current JSON
            try:
                metadata = get_video_metadata_from_s3(bucket, key)
            except FileNotFoundError:
                # If file doesn't exist, create a minimal one
                metadata = {"processing_status": "PROCESSING"}
            
            # 2. Update status
            metadata["processing_status"] = overall_status
            
            # 3. PUT updated JSON back
            S3_CLIENT.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            print(f"Successfully updated status to {overall_status} in s3://{bucket}/{key}")
            return  # Success, exit retry loop
            
        except ClientError as e:
            print(f"S3 ClientError on attempt {attempt + 1} updating status in {key}: {e}")
            if attempt == retries - 1: 
                raise  # Raise after last attempt
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            print(f"Error updating status in {key} on attempt {attempt + 1}: {e}")
            if attempt == retries - 1: 
                raise  # Raise after last attempt
            time.sleep(2 ** attempt)  # Exponential backoff

# --- Placeholder Functions for Pipeline Steps ---

def _download_video(video_url: str, local_path: str):
    print(f"Placeholder: Downloading {video_url} to {local_path}...")
    time.sleep(2) # Simulate download time
    # TODO: Implement actual download using yt-dlp or requests
    # see code in backend/technical-pipeline-scripts/download_from_url.py
    print("Placeholder: Download complete.")
    # Return path to downloaded file
    return local_path

def _upload_to_s3(local_path: str, bucket: str, s3_key: str):
    print(f"Placeholder: Uploading {local_path} to s3://{bucket}/{s3_key}...")
    # TODO: Implement S3 upload using S3_CLIENT.upload_file
    # see code in backend/upload_single_video_data.py
    time.sleep(1)
    print("Placeholder: Upload complete.")

def _chunk_video(video_s3_key: str, bucket: str) -> list:
    print(f"Placeholder: Chunking video s3://{bucket}/{video_s3_key}...")
    time.sleep(5) # Simulate chunking time
    # TODO: Implement chunking
    # see code in backend/technical-pipeline-scripts/chunk_by_scenes.py
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
     # see code in backend/technical-pipeline-scripts/caption_chunks_and_summarize.py
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
    # see code in backend/technical-pipeline-scripts/index_and_retrieve.py
    #   - Initialize Pinecone client
    #   - For each chunk with a caption:
    #     - Get caption text
    #     - Call OpenAI Embedding API
    #     - Prepare vector object (id=chunk_name, values=embedding, metadata={video_id, caption_text, start, end,...})
    #     - Upsert vectors to Pinecone in batches
    print("Placeholder: Indexing complete.")


# --- Retrieval and Context Assembly ---  (Integrated from index_and_retrieve.py)

def _retrieve_relevant_chunks(video_id: str, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Embeds the user query and retrieves relevant chunks from Pinecone,
       filtering by video_id.
    """
    print(f"Retrieving relevant chunks for video '{video_id}', query: '{user_query}'")
    if not OPENAI_CLIENT:
        print("ERROR: OpenAI client not initialized. Cannot embed query.")
        raise RuntimeError("OpenAI client not available")
    if not PINECONE_INDEX:
        print("ERROR: Pinecone index not initialized. Cannot query index.")
        raise RuntimeError("Pinecone index not available")

    embed_model = CONFIG["openai_embedding_model"]
    
    # 1. Embed the query
    try:
        start_embed = time.time()
        response = OPENAI_CLIENT.embeddings.create(
            input=[user_query],
            model=embed_model
        )
        query_vector = response.data[0].embedding
        end_embed = time.time()
        print(f"  Query embedding took: {end_embed - start_embed:.4f} seconds")
    except OpenAIError as e:
        print(f"  ERROR embedding query: {e}")
        raise RuntimeError("Failed to embed query") from e
    except Exception as e:
        print(f"  Unexpected ERROR during query embedding: {e}")
        raise

    # 2. Query Pinecone with video_id filter
    try:
        start_query = time.time()
        filter_params = {"video_id": f"{video_id}"} # Pinecone expects metadata key directly
        print(f"  Filter params: {filter_params}")
        # Note: Pinecone metadata structure in index_and_retrieve.py used video_name. 
        # Note: the video_id for our backend is <USERNAME>-<VIDEO_ID> and the video_name is structured as <USERNAME>-<VIDEO_ID>.mp4 for Pinecone
        
        query_results = PINECONE_INDEX.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_params 
        )
        end_query = time.time()
        print(f"  Pinecone query took: {end_query - start_query:.4f} seconds")

        retrieved_chunks = query_results.get('matches', [])
        print(f"  Retrieved {len(retrieved_chunks)} chunks for video '{video_id}'")
        
        # Sort by chunk_number if present
        if retrieved_chunks:
            retrieved_chunks.sort(key=lambda x: x.get('metadata', {}).get('chunk_number', float('inf')))
            print("  Sorted retrieved chunks by chunk_number.")

        return retrieved_chunks

    except PineconeException as e:
        print(f"  ERROR during Pinecone query: {e}")
        # Return empty list on Pinecone error to allow attempting context assembly
        return [] 
    except Exception as e:
        print(f"  Unexpected ERROR during Pinecone query: {e}")
        # Return empty list on general error
        return []

def _assemble_video_context(retrieved_chunks: List[Dict[str, Any]], video_metadata: Dict[str, Any]) -> str:
    """Assembles the context string from retrieved chunks and video metadata."""
    print("Assembling context...")
    
    # Extract details from the main video metadata
    video_summary = video_metadata.get("overall_summary", "No summary available.")
    video_id = video_metadata.get("video_id", "")
    # Extract TikTok user_name from video_id
    user_name = video_id.split('-')[0] if '-' in video_id else None
    key_themes = video_metadata.get("key_themes", "")
    total_duration = video_metadata.get("total_duration_seconds")
    num_chunks = video_metadata.get("num_chunks") # Might be None if not added during chunking
    num_chunks_suffix = f'/{num_chunks}' if isinstance(num_chunks, int) else ""

    context_parts = []
    context_parts.append("Video Summary:")
    context_parts.append(video_summary)

    # Add user_name to context
    if user_name:
        context_parts.append(f"\nUsername of TikTok account that posted this video:\n{user_name}")
    
    if key_themes:
        context_parts.append("\nKey Themes:")
        context_parts.append(key_themes)

    if total_duration:
        context_parts.append(f"\nTotal Video Duration: {total_duration:.2f} seconds")

    context_parts.append("\nPotentially Relevant Video Clips (in order):")
    context_parts.append("---")

    if not retrieved_chunks:
        context_parts.append("(No specific video clips retrieved based on query)")
    else:
        for i, chunk_match in enumerate(retrieved_chunks):
            metadata = chunk_match.get('metadata', {})
            seq_num = metadata.get('chunk_number', '?')
            if isinstance(seq_num, (int, float)):
                seq_num = int(seq_num)
            start_ts = metadata.get('start_timestamp', '?') # Handle missing keys gracefully
            end_ts = metadata.get('end_timestamp', '?')
            caption = metadata.get('caption', '(Caption text missing)')

            # Calculate relative time hints
            norm_start = metadata.get('normalized_start_time')
            norm_end = metadata.get('normalized_end_time')
            time_hint = ""
            hints = []
            is_valid_start = isinstance(norm_start, (float, int))
            is_valid_end = isinstance(norm_end, (float, int))

            if is_valid_start and is_valid_end:
                if norm_start <= 0.15:
                    hints.append("near the beginning")
                if norm_end >= 0.85:
                    hints.append("near the end")
                if not hints and norm_start > 0.15 and norm_end < 0.85:
                    hints.append("around the middle")
            
            if hints:
                time_hint = f" ({' and '.join(hints)})"
            
            context_parts.append(f"Video Clip {seq_num}{num_chunks_suffix} (Time: {start_ts} - {end_ts}){time_hint}:")
            context_parts.append(caption)
            if i < len(retrieved_chunks) - 1:
                context_parts.append("---")

    video_context = "\n".join(context_parts)
    print(f"Video context assembly complete. Final video context length: {len(video_context)}")
    return video_context

def _assemble_intermediate_prompt(video_context: str, query: str) -> str:
    """Assembles the intermediate prompt for our MCP Client to send to the MCP server."""
    intermediate_prompt = f"""
**Context for Query Processing:**

A user is asking a question about a video. This video context details specific observations from the video—including described entities, actions, dialogue, sounds, visuals, and overall themes. The information in video context may useful to fully and best address the user query.

---

**User Query:**
{query}

---

**Video Context:**
{video_context}
---
""" 
    return intermediate_prompt

# --- MCP Interaction & Answer Synthesis --- (Integrated from mcp_client.py)

# Rule-based tool selection logic (adapted from mcp_client.py)
def _select_perplexity_tool_rule_based(query: str) -> str:
    """Selects the appropriate Perplexity tool based on the query content using heuristics."""
    query_lower = query.lower()
    research_keywords = [
        'research', 'analyze', 'study', 'investigate', 'comprehensive', 'detailed',
        'in-depth', 'thorough', 'scholarly', 'academic', 'compare', 'contrast',
        'literature', 'history of', 'development of', 'evidence', 'sources',
        'references', 'citations', 'papers'
    ]
    deep_research_keywords = ['Deep Research', 'DeepResearch']
    reasoning_keywords = [
        'why', 'how', 'how does', 'explain', 'reasoning', 'logic', 'analyze', 'solve',
        'problem', 'prove', 'calculate', 'evaluate', 'assess', 'implications',
        'consequences', 'effects of', 'causes of', 'steps to', 'method for',
        'approach to', 'strategy', 'solution'
    ]

    is_long_query = len(query.split()) > 50
    research_score = sum(1 for keyword in research_keywords if keyword in query_lower)
    deep_research_score = sum(1 for keyword in deep_research_keywords if keyword in query_lower)
    reasoning_score = sum(1 for keyword in reasoning_keywords if keyword in query_lower)

    if is_long_query: research_score += 1
    if query_lower.startswith(('why', 'how')) and len(query_lower.split()) > 5: reasoning_score += 1

    if deep_research_score >= 1 or research_score >= 3 or (research_score >= 2 and is_long_query):
        print("  Rule-based selection: perplexity_research")
        return "perplexity_research"
    elif reasoning_score >= 2:
        print("  Rule-based selection: perplexity_reason")
        return "perplexity_reason"
    else:
        print("  Rule-based selection: perplexity_ask (default)")
        return "perplexity_ask"

# LLM-based tool selection and execution logic
async def _select_and_run_tool_llm_based(
    session: ClientSession,
    query_context: str,
    anthropic_client: Any # Should be Anthropic client instance
) -> str:
    """Uses Claude to select an MCP tool, determine args, execute it, and return the text result."""
    if not anthropic_client:
        print("  ERROR: Anthropic client not available for LLM-based tool selection.")
        return "[Error: Anthropic client not configured]"

    tool_result_text = "[LLM did not select or run a tool]" # Default if no tool use happens

    try:
        # 1. Get available tools from the MCP session
        print("  Listing tools for LLM selection...")
        list_response = await session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in list_response.tools
        ]
        if not available_tools:
            print("  Warning: No tools available from MCP server.")
            return "[Error: No tools available from MCP server]"
        print(f"  Found tools: {[t['name'] for t in available_tools]}")

        # 2. Call Anthropic API to get tool selection and arguments
        messages = [{"role": "user", "content": query_context}]
        print("  Sending query and tools to Anthropic for selection...")
        claude_response = anthropic_client.messages.create(
            model=CONFIG["anthropic_tool_selection_model"], # Use the specified model
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
            tool_choice={"type": "any"} # Force Claude to select a tool
        )

        # 3. Process Claude's response - looking specifically for tool_use
        tool_called = False
        if claude_response.content:
            for content_block in claude_response.content:
                if content_block.type == 'tool_use':
                    tool_name = content_block.name
                    tool_args = content_block.input
                    tool_use_id = content_block.id
                    print(f"  LLM selected tool: '{tool_name}' with args: {tool_args}")
                    tool_called = True

                    # 4. Execute the selected tool via MCP session
                    print(f"  Calling tool '{tool_name}' via MCP...")
                    tool_call_start = time.time()
                    try:
                        tool_exec_result = await session.call_tool(tool_name, tool_args)
                        tool_call_end = time.time()
                        print(f"  Tool call finished in {tool_call_end - tool_call_start:.2f} seconds.")

                        # 5. Extract text content from the tool execution result
                        current_tool_text = ""
                        if tool_exec_result and tool_exec_result.content:
                            for part in tool_exec_result.content:
                                if hasattr(part, 'type') and part.type == 'text' and hasattr(part, 'text'):
                                    current_tool_text += part.text + "\n"
                            tool_result_text = current_tool_text.strip()
                            print(f"  Received text result from '{tool_name}' (length: {len(tool_result_text)} chars).")
                        else:
                            print(f"  Warning: Tool '{tool_name}' returned no content.")
                            tool_result_text = f"[Tool '{tool_name}' returned no information]"

                    except Exception as e:
                        print(f"  Unexpected ERROR calling tool '{tool_name}': {e}")
                        tool_result_text = f"[Unexpected error executing tool '{tool_name}']"

                    # Assuming we only process the first tool call requested by Claude in this pass
                    break # Exit loop after handling the first tool_use block

            if not tool_called:
                 print("  LLM did not request any tool calls.")
                 # tool_result_text remains as the default message

    except AnthropicAPIError as ae:
        print(f"  ERROR: Anthropic API error during tool selection: {ae}")
        tool_result_text = f"[Error interacting with Anthropic API: {ae}]"
    except Exception as e:
        print(f"  ERROR: Unexpected error during LLM tool selection/execution: {e}")
        tool_result_text = f"[Unexpected error during LLM-based tool process: {e}]"

    return tool_result_text

async def _call_mcp(
    intermediate_prompt: str,
    mcp_server_name: str = "perplexity-ask",
    use_llm_selection: bool = True # Default to LLM-based selection
) -> str:
    """Connects to a specified MCP server (via stdio), selects and calls a tool
       (either rule-based or LLM-based), and returns the text result.
       
       WARNING: Uses stdio_client, suitable for local testing but NOT recommended for production.
    """
    print(f"Calling MCP server '{mcp_server_name}' (LLM Selection: {use_llm_selection})...")
    server_params = get_mcp_server_params(mcp_server_name)
    if not server_params:
        raise ValueError(f"MCP Server configuration '{mcp_server_name}' not found or invalid.")

    mcp_result_text = "" # Initialize

    try:
        async with AsyncExitStack() as stack:
            print(f"  Connecting to MCP server process: {server_params.command} {server_params.args}")
            # PRODUCTION NOTE: Use network client in production
            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await stack.enter_async_context(ClientSession(stdio, write))

            await session.initialize()
            print("  MCP session initialized.")

            if use_llm_selection:
                # --- LLM-Based Tool Selection --- 
                mcp_result_text = await _select_and_run_tool_llm_based(
                    session,
                    intermediate_prompt, # Pass the combined query + video context
                    ANTHROPIC_CLIENT
                )
            else:
                # --- Rule-Based Tool Selection (Existing Logic) --- 
                selected_tool = _select_perplexity_tool_rule_based(intermediate_prompt)
                print(f"  Calling tool '{selected_tool}' via MCP (rule-based selection)...")
                tool_call_start = time.time()
                try:
                    tool_args = {"messages": [{"role": "user", "content": intermediate_prompt}]}
                    result = await session.call_tool(selected_tool, tool_args)
                    tool_call_end = time.time()
                    print(f"  Tool call finished in {tool_call_end - tool_call_start:.2f} seconds.")
                    
                    # Extract text content
                    current_tool_text = ""
                    if result and result.content:
                        for part in result.content:
                            if hasattr(part, 'type') and part.type == 'text' and hasattr(part, 'text'):
                                current_tool_text += part.text + "\n"
                        mcp_result_text = current_tool_text.strip()
                        print(f"  Received text result from '{selected_tool}' (length: {len(mcp_result_text)} chars).")
                    else:
                        print(f"  Warning: Tool '{selected_tool}' returned no content.")
                        mcp_result_text = f"[Tool '{selected_tool}' returned no information]"
                except Exception as e:
                    print(f"  Unexpected ERROR calling tool '{selected_tool}': {e}")
                    mcp_result_text = f"[Unexpected error executing tool '{selected_tool}']"


    except asyncio.TimeoutError:
        print("  ERROR: Timeout connecting to or calling MCP server.")
        mcp_result_text = "[Error: Timeout interacting with MCP server]"
    except FileNotFoundError:
        print(f"  ERROR: MCP server command '{server_params.command}' not found. Is Docker running/installed?")
        mcp_result_text = f"[Error: MCP server command not found]"
    except Exception as e:
        print(f"  ERROR: Unexpected error during MCP interaction: {e}")
        mcp_result_text = f"[Unexpected error interacting with MCP server: {e}]"

    print(f"MCP call complete for server '{mcp_server_name}'.")
    return mcp_result_text

def _synthesize_answer(user_query: str, video_context: str, mcp_result: str) -> str:
    """Synthesizes the final answer using OpenAI, combining the original query,
       video context (already included in full_context), and the MCP result.
    """
    print("Synthesizing final answer using OpenAI...")
    if not OPENAI_CLIENT:
        print("ERROR: OpenAI client not initialized. Cannot synthesize answer.")
        return "[Error: OpenAI client not available for synthesis]"

    synthesis_model = CONFIG.get("openai_synthesis_model", "gpt-4o-mini")

    # Construct the prompt for the synthesis model
    # The `full_context` variable already contains the User Query, Video Summary, Themes, and Relevant Clips.
    # We just need to add the MCP result as Additional Information.
    # Modify this prompt structure as needed for optimal results.
    prompt = f"""
**Task:**
Please answer the user query comprehensively by synthesizing relevant information from **both** the Video Context (details extracted directly from the video) and the relevant Internet Search Results provided below.

**Instructions:**
1.  Analyze the User Query embedded within the Video Context to understand the core question.
2.  Review the Video Context (summary, themes, specific segments) for information directly observable in the video.
3.  Review the Internet Search Results for broader context, facts, or related information.
4.  Formulate a cohesive answer that integrates relevant details from both sources.
5.  Prioritize information from the Video Context when the query pertains to specific events or details *within* the video itself.
6.  Use the Internet Search Results to enrich the answer, provide background, clarify concepts, or address aspects of the query not covered by the video context alone.
7.  If the combined information is insufficient to answer the query fully, state what information is available and what is missing. Do not speculate beyond the provided contexts.
8.  Provide a clear and concise answer. Do not include citations.

---

**User Query:**
{user_query}

---

**Video Context (Includes User Query):**
{video_context}

---

**Internet Search Results (from Perplexity):**
{mcp_result}

---

**Final Answer:**
"""

    print(f"  Sending synthesis prompt to OpenAI model: {synthesis_model}")
    synthesis_start = time.time()
    try:
        completion = OPENAI_CLIENT.chat.completions.create(
            model=synthesis_model,
            messages=[
                # Note: Providing the entire combined text as a single user message.
                # You could experiment with different roles or structuring if needed.
                {"role": "user", "content": prompt}
            ]
        )
        final_answer = completion.choices[0].message.content
        synthesis_end = time.time()
        print(f"  OpenAI synthesis successful in {synthesis_end - synthesis_start:.2f} seconds.")
        return final_answer.strip() if final_answer else "[OpenAI returned an empty answer]"

    except OpenAIError as e:
        print(f"  ERROR during OpenAI synthesis: {e}")
        return f"[Error synthesizing answer using OpenAI: {e}]"
    except Exception as e:
        print(f"  Unexpected ERROR during OpenAI synthesis: {e}")
        return f"[Unexpected error during answer synthesis: {e}]"



# --- Background Task Implementations --- (Updated)

async def run_query_pipeline_async(video_id: str, user_query: str, interaction_id: str, query_timestamp: str, s3_json_path: str, s3_interactions_path: str, s3_bucket: str):
    """Background task to answer a query for a PROCESSED video."""
    print(f"BACKGROUND TASK: Starting query pipeline for interaction {interaction_id} on video {video_id}")
    start_time = time.time()

    try:
        # 1. Add interaction with "processing" status
        interaction = {
            "interaction_id": interaction_id,
            "user_query": user_query,
            "query_timestamp": query_timestamp,
            "status": "processing"
        }
        add_interaction_to_s3(s3_bucket, s3_interactions_path, interaction)

        # 2. Load full video metadata
        video_metadata = get_video_metadata_from_s3(s3_bucket, s3_json_path)
        print(f"===============\nVIDEO METADATA LOADED\n===============")

        # 3. Retrieve relevant chunks from Pinecone
        retrieved_chunks = _retrieve_relevant_chunks(video_id, user_query)
        print(f"===============\nRETRIEVED {len(retrieved_chunks)} CHUNKS\n===============")

        # 4. Assemble context (This now includes summary, themes, clips)
        # Prepend the original user query for clarity in the combined context
        video_context = _assemble_video_context(retrieved_chunks, video_metadata)
        intermediate_prompt = _assemble_intermediate_prompt(video_context, user_query)
        print(f"===============\nINTERMEDIATE PROMPT:\n{intermediate_prompt[:500]}...\n===============")
        
        # 5. Call MCP tool (using the assembled context + query)
        # Decide here whether to use LLM selection or rule-based
        # For now, let's keep the default (use_llm_selection=False in _call_mcp)
        mcp_result = await _call_mcp(intermediate_prompt) 
        print(f"===============\nMCP TOOL RESULT:\n{mcp_result}\n===============")
        
        # 6. Synthesize final answer (using the original context, MCP result, and query)
        final_answer = _synthesize_answer(user_query, video_context, mcp_result)
        print(f"===============\nFINAL ANSWER:\n{final_answer[:500]}...\n===============")
        
        # 7. Update status to completed with the answer
        update_interaction_status_in_s3(
            s3_bucket, s3_interactions_path, interaction_id, "completed",
            final_answer=final_answer,
            answer_timestamp=datetime.now(timezone.utc).isoformat()
        )
        print(f"BACKGROUND TASK: Query pipeline for interaction {interaction_id} COMPLETED.")

    except Exception as e:
        print(f"BACKGROUND TASK ERROR: Query pipeline for interaction {interaction_id} FAILED: {e}")
        try:
            # Attempt to mark as failed
            update_interaction_status_in_s3(s3_bucket, s3_interactions_path, interaction_id, "failed")
        except Exception as update_e:
            print(f"BACKGROUND TASK ERROR: Failed to update status to failed for {interaction_id}: {update_e}")
    finally:
        end_time = time.time()
        print(f"BACKGROUND TASK: Query pipeline for interaction {interaction_id} finished in {end_time - start_time:.2f} seconds.")


async def run_full_pipeline_async(video_url: str, user_query: str, video_id: str, s3_video_base_path: str, s3_json_path: str, s3_interactions_path: str, s3_bucket: str, interaction_id: str, query_timestamp: str):
    """Background task to process a NEW video and then answer a query."""
    print(f"BACKGROUND TASK: Starting full pipeline for {video_id} with interaction {interaction_id}")
    start_time = time.time()
    full_video_metadata = None

    try:
        # 1. Create initial metadata file
        initial_metadata = {
            "video_id": video_id,
            "source_url": video_url,
            "processing_status": "PROCESSING",
            "processing_start_time": datetime.now(timezone.utc).isoformat()
        }
        S3_CLIENT.put_object(
            Bucket=s3_bucket,
            Key=s3_json_path,
            Body=json.dumps(initial_metadata, indent=2),
            ContentType='application/json'
        )
        print(f"Created initial metadata at s3://{s3_bucket}/{s3_json_path}")
        
        # 2. Add initial interaction
        interaction = {
            "interaction_id": interaction_id,
            "user_query": user_query,
            "query_timestamp": query_timestamp,
            "status": "processing"
        }
        add_interaction_to_s3(s3_bucket, s3_interactions_path, interaction)

        # --- Video Processing Steps ---
        local_video_path = _download_video(video_url, f"/tmp/{video_id}.mp4") # TODO: Handle /tmp path robustly
        _upload_to_s3(local_video_path, s3_bucket, f"{s3_video_base_path}.mp4")
        chunks_metadata = _chunk_video(f"{s3_video_base_path}.mp4", s3_bucket)
        chunks_with_captions, overall_summary, key_themes = _generate_captions_and_summary(
            chunks_metadata, s3_video_base_path, s3_bucket
        )
        _index_captions(video_id, chunks_with_captions)

        # --- Finalize Metadata ---
        # This is the complete metadata after processing
        full_video_metadata = {
            "video_id": video_id,
            "source_url": video_url,
            "processing_status": "FINISHED", # Mark as finished *before* query handling
            "processing_complete_time": datetime.now(timezone.utc).isoformat(),
            "overall_summary": overall_summary,
            "key_themes": key_themes,
            "chunks": chunks_with_captions,
            # TODO: Consider adding total_duration and num_chunks here if available from chunking step
        }
        S3_CLIENT.put_object(
            Bucket=s3_bucket,
            Key=s3_json_path,
            Body=json.dumps(full_video_metadata, indent=2),
            ContentType='application/json'
        )
        print(f"Updated metadata with status FINISHED at s3://{s3_bucket}/{s3_json_path}")

        # --- Handle the Query --- 
        # Retrieve relevant chunks
        retrieved_chunks = _retrieve_relevant_chunks(video_id, user_query)
        
        # Assemble video context and intermediate prompt
        video_context = _assemble_video_context(retrieved_chunks, full_video_metadata)
        intermediate_prompt = _assemble_intermediate_prompt(video_context, user_query)
        print(f"Generated intermediate prompt (length: {len(intermediate_prompt)} chars)")
        
        # Call MCP (using default rule-based selection for now)
        mcp_result = await _call_mcp(intermediate_prompt) 
        print(f"Received MCP result (length: {len(mcp_result)} chars)")
        
        # Synthesize answer
        final_answer = _synthesize_answer(user_query, video_context, mcp_result)
        print(f"Synthesized final answer (length: {len(final_answer)} chars)")
        
        # Update interaction status to completed
        update_interaction_status_in_s3(
            s3_bucket, s3_interactions_path, interaction_id, "completed",
            final_answer=final_answer,
            answer_timestamp=datetime.now(timezone.utc).isoformat()
        )
        print(f"BACKGROUND TASK: Full pipeline for {video_id} with interaction {interaction_id} COMPLETED.")

    except Exception as e:
        print(f"BACKGROUND TASK ERROR: Full pipeline for {video_id} with interaction {interaction_id} FAILED: {e}")
        try:
            # Mark as failed in both files if possible
            if full_video_metadata: # Check if metadata was created before failing
                update_overall_processing_status(s3_bucket, s3_json_path, "FAILED")
            else:
                print("Skipping overall status update to FAILED as initial metadata might not exist.")
            # Always try to update interaction status
            update_interaction_status_in_s3(s3_bucket, s3_interactions_path, interaction_id, "failed")
        except Exception as update_e:
            print(f"BACKGROUND TASK ERROR: Failed to update status to failed: {update_e}")
    finally:
        end_time = time.time()
        print(f"BACKGROUND TASK: Full pipeline for {video_id} finished in {end_time - start_time:.2f} seconds.")

