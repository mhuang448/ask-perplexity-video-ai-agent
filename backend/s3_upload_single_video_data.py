import time
import boto3
import os
import sys
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

def upload_video_structure(s3_client, local_video_dir, bucket_name, s3_base_prefix="processed-videos/"):
    """
    Uploads a specific video's directory structure to S3 using a provided client.

    - All uploaded files will inherit the bucket's default object ownership (likely private).
    - Logs success or failure for each file upload attempt.

    Args:
        s3_client: An initialized boto3 S3 client instance.
        local_video_dir (str): Path to the local directory for the video (e.g., './processed-videos/<USER_NAME>-<VIDEO_ID>').
        bucket_name (str): Name of the target S3 bucket.
        s3_base_prefix (str): The base prefix in S3 (e.g., 'processed-videos/'). Must end with '/'.
    """
    # Check if directory exists
    if not os.path.isdir(local_video_dir):
        print(f"Error: Local directory '{local_video_dir}' not found. Skipping.")
        return False # Indicate failure for this directory

    video_id = os.path.basename(local_video_dir) # Get '<USER_NAME>-<VIDEO_ID>' from the path
    s3_prefix = f"{s3_base_prefix}{video_id}/" # Target S3 prefix, e.g., "processed-videos/<USER_NAME>-<VIDEO_ID>/"
    success = True # Track overall success for this video directory

    print(f"Processing '{video_id}' -> s3://{bucket_name}/{s3_prefix}")

    # Iterate through items in the local video directory
    for item_name in os.listdir(local_video_dir):
        local_item_path = os.path.join(local_video_dir, item_name)
        s3_key = f"{s3_prefix}{item_name}"

        # --- Handle Main MP4 File ---
        if item_name == f"{video_id}.mp4" and os.path.isfile(local_item_path):
            print(f"  Uploading main video '{item_name}' to '{s3_key}' (private)...")
            try:
                s3_client.upload_file(
                    local_item_path,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'video/mp4'} # ACL removed
                )
                print(f"  SUCCESS: Uploaded '{item_name}' to '{s3_key}'.")
            except ClientError as e:
                print(f"  ERROR uploading {item_name}: {e}")
                success = False # Mark failure
            continue # Move to next item

        # --- Handle Main JSON File ---
        if item_name == f"{video_id}.json" and os.path.isfile(local_item_path):
            print(f"  Uploading metadata '{item_name}' to '{s3_key}' (private)...")
            try:
                s3_client.upload_file(
                    local_item_path,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'application/json'}
                    # No ACL specified = private
                )
                print(f"  SUCCESS: Uploaded '{item_name}' to '{s3_key}'.")
            except ClientError as e:
                print(f"  ERROR uploading {item_name}: {e}")
                success = False # Mark failure
            continue # Move to next item

        # --- Handle Chunks Directory ---
        if item_name == "chunks" and os.path.isdir(local_item_path):
            print(f"  Processing 'chunks' directory...")
            chunks_s3_prefix = f"{s3_prefix}chunks/" # e.g., processed-videos/<USER_NAME>-<VIDEO_ID>/chunks/
            for chunk_filename in os.listdir(local_item_path):
                local_chunk_path = os.path.join(local_item_path, chunk_filename)
                if os.path.isfile(local_chunk_path) and chunk_filename.lower().endswith('.mp4'):
                    chunk_s3_key = f"{chunks_s3_prefix}{chunk_filename}"
                    print(f"    Uploading chunk '{chunk_filename}' to '{chunk_s3_key}' (private)...")
                    try:
                        s3_client.upload_file(
                            local_chunk_path,
                            bucket_name,
                            chunk_s3_key,
                            ExtraArgs={'ContentType': 'video/mp4'}
                            # No ACL specified = private
                        )
                        print(f"    SUCCESS: Uploaded chunk '{chunk_filename}' to '{chunk_s3_key}'.")
                    except ClientError as e:
                        print(f"    ERROR uploading chunk {chunk_filename}: {e}")
                        success = False # Mark failure
                elif os.path.isfile(local_chunk_path):
                    print(f"    Skipping non-mp4 file in chunks: {chunk_filename}")
                # Implicitly skips sub-directories within chunks
            continue # Move to next item

        # --- Handle other items (like .DS_Store) ---
        if os.path.isfile(local_item_path):
             print(f"  Skipping unknown file: {item_name}")
        elif os.path.isdir(local_item_path):
             print(f"  Skipping unknown sub-directory: {item_name}")
        # Add handling for other potential item types if needed


    print(f"Finished processing '{video_id}'. Overall success: {success}")
    return success


if __name__ == "__main__":
    time_start = time.time()
    if len(sys.argv) != 2:
        print("Usage: python s3_upload_single_video_data.py <path_to_local_video_dir>")
        print("Example: python s3_upload_single_video_data.py ./processed-videos/aichifan33-7486040114695507242")
        sys.exit(1)

    local_dir = sys.argv[1]
    bucket = os.getenv("S3_BUCKET_NAME")

    upload_video_structure(local_dir, bucket)

    time_end = time.time()
    print(f"Total upload time for {local_dir}:\n{time_end - time_start} seconds")