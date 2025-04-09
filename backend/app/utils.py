# app/utils.py
import os
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
        s3_client.list_buckets()
        print("S3 Client Initialized Successfully.")
        return s3_client
    except (NoCredentialsError, PartialCredentialsError):
        print("ERROR: AWS credentials not found. Configure AWS CLI, IAM Role, or environment variables.")
        raise
    except Exception as e:
        print(f"ERROR: Failed to initialize S3 client: {e}")
        raise

S3_CLIENT = get_s3_client() # Initialize client once

# --- Helper Functions ---

def generate_unique_video_id(url: str) -> str:
    """Generates a somewhat unique ID based on the URL (improve as needed)."""
    # Simple approach for PoC - consider hashing for better uniqueness
    base_id = str(uuid.uuid4()) # Start with a random UUID
    # You could try extracting parts of the URL if they are reliably unique
    # url_parts = url.split('/')
    # if len(url_parts) > 4:
    #    base_id = f"{url_parts[3]}_{url_parts[5]}"
    print(f"Generated video ID: {base_id} for URL: {url}")
    return base_id

PREPROCESSED_PREFIX = "preprocessed/"
PROCESSED_PREFIX = "processed/"

def get_s3_json_path(video_id: str, is_preprocessed: bool) -> str:
    """Constructs the S3 key (path) for the video's JSON metadata file."""
    prefix = PREPROCESSED_PREFIX if is_preprocessed else PROCESSED_PREFIX
    return f"{prefix}{video_id}/{video_id}.json"

def get_s3_video_base_path(video_id: str, is_preprocessed: bool) -> str:
    """Constructs the base S3 key (path) for video/chunk files."""
    prefix = PREPROCESSED_PREFIX if is_preprocessed else PROCESSED_PREFIX
    return f"{prefix}{video_id}/{video_id}" # e.g., processed/vid123/vid123 -> .mp4 or /chunks/

def determine_if_preprocessed(video_id: str) -> bool:
    """Placeholder logic to guess if a video ID is preprocessed."""
    # TODO: Implement real logic. Maybe check if the JSON exists under PREPROCESSED_PREFIX first?
    # Or rely on a naming convention for generated IDs vs preprocessed IDs.
    print(f"Warning: Using placeholder logic for determine_if_preprocessed for {video_id}")
    # Simplistic guess: If it looks like a UUID, maybe it's newly generated?
    try:
        uuid.UUID(video_id)
        return False # Looks like a UUID, assume newly processed
    except ValueError:
        return True # Doesn't look like a UUID, assume preprocessed
