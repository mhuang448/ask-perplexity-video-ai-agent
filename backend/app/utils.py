# app/utils.py
import os
import re
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv
import uuid
from typing import Optional, Dict, Any

# Load .env file ONLY for local development.
# In production (App Runner), environment variables are set directly.
load_dotenv()

# --- Configuration Loading ---

def load_config() -> Dict[str, Optional[str]]:
    """Loads configuration from environment variables."""
    config = {
        "aws_region": os.getenv("AWS_REGION", "us-east-2"),
        "s3_bucket_name": os.getenv("S3_BUCKET_NAME"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_index_host": os.getenv("PINECONE_INDEX_HOST"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"), # Or handle GOOGLE_APPLICATION_CREDENTIALS
        "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        # Add direct AWS keys only if absolutely needed (prefer IAM roles)
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }
    # Basic validation (add more as needed)
    if not config["s3_bucket_name"]:
        raise ValueError("Missing required environment variable: S3_BUCKET_NAME")
    # Add checks for other essential keys...
    return config

CONFIG = load_config() # Load config once when the module is imported

# --- AWS S3 Client Setup ---

def get_s3_client():
    """Initializes and returns an S3 client."""
    try:
        # If AWS keys are set in ENV, boto3 uses them.
        # If running on EC2/App Runner with an IAM role, boto3 uses the role automatically.
        # If AWS CLI is configured locally, boto3 uses those credentials.
        s3_client = boto3.client(
            's3',
            region_name=CONFIG["aws_region"],
            # Explicitly pass keys only if absolutely necessary (not recommended)
            # aws_access_key_id=CONFIG["aws_access_key_id"],
            # aws_secret_access_key=CONFIG["aws_secret_access_key"],
        )
        # Test connection briefly (optional)
        buckets = s3_client.list_buckets()['Buckets']
        print(f"S3 Client Initialized Successfully. Buckets: {buckets}")
        return s3_client
    except (NoCredentialsError, PartialCredentialsError):
        print("ERROR: AWS credentials not found. Configure AWS CLI, IAM Role, or environment variables.")
        raise
    except Exception as e:
        print(f"ERROR: Failed to initialize S3 client: {e}")
        raise

S3_CLIENT = get_s3_client() # Initialize client once

# --- Helper Functions ---

def generate_unique_video_name(url: str) -> str:
    """Generates a unique name based on the URL (improve as needed). 
    To be used as primary ID of video in videos/ directory in S3."""
    # # Simple approach for PoC - consider hashing for better uniqueness
    # base_id = str(uuid.uuid4()) # Start with a random UUID
    # print(f"Generated video ID: {base_id} for URL: {url}")
    match = re.search(r"@(?P<username>[^/]+)/video/(?P<video_id>\d+)", url)
    if not match:
        print(f"Error: Could not extract username and video ID from URL: {url}")
        return

    username = match.group("username")
    video_id = match.group("video_id")

    # Construct the name: username-videoid
    video_name = f"{username}-{video_id}"
    return video_name

VIDEO_DATA_PREFIX = "video-data/"

def get_s3_json_path(video_name: str) -> str:
    """Constructs the S3 key (path) for the video's JSON metadata file."""
    return f"{VIDEO_DATA_PREFIX}{video_name}/{video_name}.json"

def get_s3_interactions_path(video_name: str) -> str:
    """Constructs the S3 key (path) for the video's interactions.json file.
    This file is separate from the main metadata to reduce concurrency issues."""
    return f"{VIDEO_DATA_PREFIX}{video_name}/interactions.json"

def get_s3_video_base_path(video_name: str) -> str:
    """Constructs the base S3 key (path) for video/chunk files."""
    return f"{VIDEO_DATA_PREFIX}{video_name}/{video_name}" # e.g., video-data/<USER_NAME>-<VIDEO_ID>/<USER_NAME>-<VIDEO_ID> -> .mp4 or /chunks/

def determine_if_processed(video_name: str) -> bool:
    """Placeholder function to determine if a video is processed.
    In a real implementation, this might check for the existence of the file
    or review flags in a database.
    """
    # For PoC, we assume this is a newly processed video (not processed)
    # You could implement actual logic here if needed
    return False
