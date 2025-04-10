# app/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import random
import uuid
from datetime import datetime, timezone

# Import models, utils, and pipeline logic
from .models import (
    QueryRequest, ProcessRequest, VideoInfo,
    ProcessingStartedResponse, StatusResponse
)
from .utils import (
    CONFIG, determine_if_processed, get_s3_json_path, get_s3_interactions_path,
    generate_unique_video_name, get_s3_video_base_path
)
from .pipeline_logic import (
    run_query_pipeline_async, run_full_pipeline_async,
    get_processing_status_from_s3, get_interactions_from_s3,
    # Assume this helper exists to list preprocessed video IDs
    # get_list_of_preprocessed_video_ids
)

# --- FastAPI App Setup ---
app = FastAPI(
    title="Video Q&A AI Agent API",
    description="API for processing videos and answering questions using RAG+MCP.",
    version="0.1.0"
)

# --- CORS Configuration ---
# Define allowed origins (replace with your actual frontend URLs)
origins = [
    "http://localhost:3000", # Local Next.js frontend
    # Add your Vercel deployment URL(s) here after deployment
    # e.g., "https://your-frontend-app-name.vercel.app",
    # "*" # Allow all origins (less secure, use specific origins in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows GET, POST, etc.
    allow_headers=["*"],
)

# --- Placeholder for listing preprocessed videos ---
# TODO: Implement this properly, e.g., by listing S3 keys or reading config
def get_list_of_preprocessed_video_ids_stub() -> list[str]:
    print("Warning: Using hardcoded list of processed video names")
    return [f"processed_video_{i}" for i in range(1, 11)] # Example names


# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
async def read_root():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Video Q&A AI Agent API!"}


@app.get("/api/videos/foryou", response_model=list[VideoInfo], tags=["Videos"])
async def get_for_you_videos():
    """Returns a list of 3 random pre-processed video IDs and public S3 URLs."""
    try:
        # Replace stub with actual implementation later
        all_preprocessed_ids = get_list_of_preprocessed_video_ids_stub()
        if not all_preprocessed_ids:
            return []

        sample_size = min(len(all_preprocessed_ids), 3)
        selected_ids = random.sample(all_preprocessed_ids, sample_size)

        s3_bucket = CONFIG["s3_bucket_name"]
        videos = []
        for video_id in selected_ids:
            # Construct the public S3 URL (ensure bucket/objects are public and CORS enabled on S3)
            # Note: Using virtual-hosted style URL (bucket name in domain)
            video_url = f"https://{s3_bucket}.s3.{CONFIG['aws_region']}.amazonaws.com/{get_s3_video_base_path(video_id)}.mp4"
            videos.append(VideoInfo(video_id=video_id, video_url=video_url))

        return videos
    except Exception as e:
        print(f"Error in /api/videos/foryou: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve videos.")


@app.post("/api/query/async", response_model=ProcessingStartedResponse, status_code=202, tags=["Query"])
async def query_preprocessed_video(query: QueryRequest, background_tasks: BackgroundTasks):
    """Triggers async query processing for a PRE-PROCESSED video."""
    print(f"Received query for preprocessed video {query.video_id}: {query.user_query}")

    interaction_id = str(uuid.uuid4())
    query_timestamp = datetime.now(timezone.utc).isoformat()
    s3_json_path = get_s3_json_path(query.video_id)
    s3_interactions_path = get_s3_interactions_path(query.video_id)
    s3_bucket = CONFIG["s3_bucket_name"]

    # Create the interaction data structure
    interaction = {
        "interaction_id": interaction_id,
        "user_query": query.user_query,
        "query_timestamp": query_timestamp,
        "status": "processing"
    }

    background_tasks.add_task(
        run_query_pipeline_async,
        query.video_id,
        query.user_query,
        interaction_id,
        query_timestamp,
        s3_json_path,
        s3_interactions_path,
        s3_bucket
    )

    return ProcessingStartedResponse(
        status="Query processing started",
        video_id=query.video_id,
        interaction_id=interaction_id
    )


@app.post("/api/process_and_query/async", response_model=ProcessingStartedResponse, status_code=202, tags=["Query"])
async def process_new_video_and_query(process_req: ProcessRequest, background_tasks: BackgroundTasks):
    """Triggers async FULL pipeline (download to answer) for a NEW video URL."""
    print(f"Received request for new video {process_req.video_url} with query: {process_req.user_query}")

    video_id = generate_unique_video_name(process_req.video_url)
    interaction_id = str(uuid.uuid4())
    query_timestamp = datetime.now(timezone.utc).isoformat()

    s3_json_path = get_s3_json_path(video_id)
    s3_interactions_path = get_s3_interactions_path(video_id)
    s3_video_base_path = get_s3_video_base_path(video_id)
    s3_bucket = CONFIG["s3_bucket_name"]

    # Create the interaction data structure
    interaction = {
        "interaction_id": interaction_id,
        "user_query": process_req.user_query,
        "query_timestamp": query_timestamp,
        "status": "processing"
    }

    background_tasks.add_task(
        run_full_pipeline_async,
        process_req.video_url,
        process_req.user_query,
        video_id,
        s3_video_base_path,
        s3_json_path,
        s3_interactions_path,
        s3_bucket,
        interaction_id,
        query_timestamp
    )

    return ProcessingStartedResponse(
        status="Full video processing and query started",
        video_id=video_id,
        interaction_id=interaction_id
    )


@app.get("/api/query/status/{video_id}", response_model=StatusResponse, tags=["Query"])
async def get_query_status(video_id: str):
    """Pollable endpoint to check status and get all interactions from S3 JSON."""
    print(f"Checking status for video_id: {video_id}")
    s3_bucket = CONFIG["s3_bucket_name"]

    # Get paths for both files
    s3_json_path = get_s3_json_path(video_id)
    s3_interactions_path = get_s3_interactions_path(video_id)
    
    try:
        # Get video metadata for processing_status
        video_metadata = get_processing_status_from_s3(s3_bucket, s3_json_path)
        processing_status = video_metadata.get("processing_status")
        
        # Get interactions (may not exist yet)
        try:
            interactions = get_interactions_from_s3(s3_bucket, s3_interactions_path)
        except Exception as e:
            print(f"Note: Could not retrieve interactions (might not exist yet): {e}")
            interactions = []
        
        # Ensure response matches the Pydantic model
        return StatusResponse(
            video_id=video_id,
            processing_status=processing_status,
            interactions=interactions
        )
    except FileNotFoundError:
        print(f"Metadata file not found for {video_id} at {s3_json_path}")
        raise HTTPException(status_code=404, detail=f"Status not available for video {video_id}. Processing may not have started or completed initial steps.")
    except Exception as e:
        print(f"Error getting status for {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve status.")

# --- Optional: Run directly for local testing (though `uvicorn main:app --reload` is better) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
