# app/utils.py
import os
import re
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from dotenv import load_dotenv
import uuid
from typing import Optional, Dict, Any
import time # Added for Pinecone index readiness check
# Added imports for OpenAI and Pinecone
from openai import OpenAI, OpenAIError
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone.exceptions import PineconeException

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
        "pinecone_index_host": os.getenv("PINECONE_INDEX_HOST"), # May be set dynamically if not present
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME", "video-captions-index"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
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
    if not config["pinecone_api_key"]:
        print("Warning: Missing PINECONE_API_KEY environment variable.")
    if not config["openai_api_key"]:
        print("Warning: Missing OPENAI_API_KEY environment variable.")
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
        # Check if bucket exists to validate credentials/permissions early
        s3_client.head_bucket(Bucket=CONFIG["s3_bucket_name"])
        print(f"S3 Client Initialized Successfully for bucket '{CONFIG['s3_bucket_name']}'.")
        return s3_client
    except (NoCredentialsError, PartialCredentialsError):
        print("ERROR: AWS credentials not found. Configure AWS CLI, IAM Role, or environment variables.")
        raise
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"ERROR: S3 bucket '{CONFIG['s3_bucket_name']}' not found or access denied.")
        else:
            print(f"ERROR: Failed to initialize S3 client: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error initializing S3 client: {e}")
        raise

S3_CLIENT = get_s3_client() # Initialize client once

# --- OpenAI Client Setup ---
def get_openai_client():
    """Initializes and returns an OpenAI client."""
    if not CONFIG["openai_api_key"]:
        print("ERROR: OpenAI API key not configured.")
        return None # Allow graceful failure if not needed immediately
    try:
        client = OpenAI(api_key=CONFIG["openai_api_key"])
        # Optional: Make a simple test call, e.g., list models
        # client.models.list()
        print("OpenAI Client Initialized Successfully.")
        return client
    except OpenAIError as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error initializing OpenAI client: {e}")
        raise

OPENAI_CLIENT = get_openai_client()

# --- Pinecone Client Setup ---
def get_pinecone_client_and_index():
    """Initializes Pinecone client and connects to the specified index.
       Handles dynamic host discovery if PINECONE_INDEX_HOST is not set.
    """
    if not CONFIG["pinecone_api_key"]:
        print("ERROR: Pinecone API key not configured.")
        return None, None # Allow graceful failure

    index_name = CONFIG["pinecone_index_name"]
    index_host = CONFIG["pinecone_index_host"]

    try:
        print("Initializing Pinecone client...")
        pc = Pinecone(api_key=CONFIG["pinecone_api_key"])

        if not index_host:
            print(f"PINECONE_INDEX_HOST not set. Attempting to discover host for index '{index_name}'...")
            try:
                existing_indexes = pc.list_indexes().names
                if index_name not in existing_indexes:
                    # In a production scenario, we might not want the backend to create the index.
                    # This assumes the index should already exist.
                    print(f"ERROR: Pinecone index '{index_name}' not found.")
                    raise ValueError(f"Pinecone index '{index_name}' does not exist.")

                print(f"Describing index '{index_name}' to get host...")
                index_description = pc.describe_index(index_name)

                # Check if index is ready
                max_wait_time = 60
                wait_start_time = time.time()
                while not index_description.status['ready']:
                    if time.time() - wait_start_time > max_wait_time:
                         raise TimeoutError(f"Index '{index_name}' did not become ready within {max_wait_time} seconds.")
                    print("Index not ready yet, waiting 5 seconds...")
                    time.sleep(5)
                    index_description = pc.describe_index(index_name)

                index_host = index_description.host
                CONFIG["pinecone_index_host"] = index_host # Update config for this session
                print(f"Discovered and using Pinecone index host: {index_host}")
                # Note: This does not update the .env file. Host should ideally be set permanently.
            except PineconeException as e:
                print(f"ERROR: Pinecone API error during host discovery: {e}")
                raise
            except Exception as e:
                print(f"ERROR: Unexpected error during Pinecone host discovery: {e}")
                raise

        print(f"Connecting to Pinecone index '{index_name}' via host: {index_host}")
        pinecone_index = pc.Index(host=index_host)
        # Test connection with describe_index_stats
        stats = pinecone_index.describe_index_stats()
        print(f"Pinecone Client and Index '{index_name}' Initialized Successfully. Stats: {stats}")
        return pc, pinecone_index

    except PineconeException as e:
        print(f"ERROR: Failed to initialize Pinecone client or index: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error initializing Pinecone client or index: {e}")
        raise

PINECONE_CLIENT, PINECONE_INDEX = get_pinecone_client_and_index()

# --- Helper Functions ---

def generate_unique_video_id(url: str) -> str:
    """Generates a unique id based on the URL (TikTok format assumed)."""
    match = re.search(r"@(?P<username>[^/]+)/video/(?P<video_id>\d+)", url)
    if not match:
        print(f"Warning: Could not extract username/video_id from URL: {url}. Generating UUID-based name.")
        # Fallback to UUID if pattern doesn't match
        return str(uuid.uuid4())

    username = match.group("username")
    tiktok_video_id = match.group("video_id")
    video_id = f"{username}-{tiktok_video_id}"
    print(f"Generated video_id: {video_id} from URL: {url}")
    return video_id

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
