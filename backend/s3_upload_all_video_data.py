import boto3
import os
import sys
import time # Added for potential retries/logging
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
                upload_start_time = time.time()
                s3_client.upload_file(
                    local_item_path,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'video/mp4'} # ACL removed
                )
                upload_end_time = time.time()
                print(f"  SUCCESS: Uploaded '{item_name}' to '{s3_key}' in {upload_end_time - upload_start_time:.2f} seconds.")
            except ClientError as e:
                print(f"  ERROR uploading {item_name}: {e}")
                success = False # Mark failure
            continue # Move to next item

        # --- Handle Main JSON File ---
        if item_name == f"{video_id}.json" and os.path.isfile(local_item_path):
            print(f"  Uploading metadata '{item_name}' to '{s3_key}' (private)...")
            try:
                upload_start_time = time.time()
                s3_client.upload_file(
                    local_item_path,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'application/json'}
                    # No ACL specified = private
                )
                upload_end_time = time.time()
                print(f"  SUCCESS: Uploaded '{item_name}' to '{s3_key}' in {upload_end_time - upload_start_time:.2f} seconds.")
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
                        upload_start_time = time.time()
                        s3_client.upload_file(
                            local_chunk_path,
                            bucket_name,
                            chunk_s3_key,
                            ExtraArgs={'ContentType': 'video/mp4'}
                            # No ACL specified = private
                        )
                        upload_end_time = time.time()
                        print(f"    SUCCESS: Uploaded chunk '{chunk_filename}' to '{chunk_s3_key}' in {upload_end_time - upload_start_time:.2f} seconds.")
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


def upload_all_processed(local_base_dir, bucket_name, s3_target_prefix="processed-videos/"):
    """
    Iterates through subdirectories in local_base_dir and uploads each video structure to S3.
    """
    start_time = time.time()
    print(f"Starting bulk upload from '{local_base_dir}' to 's3://{bucket_name}/{s3_target_prefix}'")
    processed_count = 0
    success_count = 0
    failed_dirs = []

    try:
        s3_client = boto3.client('s3', region_name=os.getenv("AWS_REGION"))
        # Verify connection once (optional but good practice)
        print("Attempting to verify AWS credentials and region...")
        s3_client.list_buckets() # Simple operation to test credentials
        print("AWS credentials and region verified.")

        if not os.path.isdir(local_base_dir):
            print(f"Error: Local base directory '{local_base_dir}' not found.")
            return

        # List items in the base directory
        dir_items = sorted(os.listdir(local_base_dir)) # Sort for consistent processing order
        for item_name in dir_items:
            potential_video_dir = os.path.join(local_base_dir, item_name)
            # Check if the item is a directory (representing a single video's folder)
            if os.path.isdir(potential_video_dir):
                processed_count += 1
                try:
                    # Pass the single s3_client instance
                    if upload_video_structure(s3_client, potential_video_dir, bucket_name, s3_target_prefix):
                        success_count += 1
                    else:
                        failed_dirs.append(item_name)
                except Exception as e:
                    # Catch unexpected errors during a specific directory's processing
                    print(f"!! UNEXPECTED ERROR processing directory '{item_name}': {e}")
                    failed_dirs.append(item_name)
            else:
                # Optionally log skipped files/items
                if item_name != ".DS_Store": # Common macOS file to ignore silently
                    print(f"Skipping item (not a directory): {item_name}")

        end_time = time.time()
        print("-" * 30)
        print(f"Bulk upload finished.")
        print(f"  Processed: {processed_count} potential video directories.")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {len(failed_dirs)}")
        if failed_dirs:
            print(f"  Failed directories: {', '.join(failed_dirs)}")
        print(f"  Total time: {end_time - start_time:.2f} seconds.")

    except NoCredentialsError:
        print("ERROR: AWS credentials not found. Configure via ~/.aws/credentials, environment variables, or an IAM role.")
    except ClientError as e:
         # Handle errors occurring during the initial client setup/verification
         error_code = e.response.get('Error', {}).get('Code')
         if error_code == 'AccessDenied':
             print(f"ERROR: Access Denied verifying credentials (check IAM permissions, e.g., s3:ListAllMyBuckets might be needed for verification step).")
         elif error_code == 'InvalidAccessKeyId':
              print(f"ERROR: Invalid AWS Access Key ID. Please check your credentials.")
         elif error_code == 'SignatureDoesNotMatch':
              print(f"ERROR: AWS Signature mismatch. Please check your Secret Access Key and region.")
         else:
             print(f"AWS ClientError during setup/verification: {e}")
    except Exception as e:
        # Catch other unexpected errors during setup or listing
        print(f"An unexpected error occurred during initialization or directory listing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python s3_upload_all_video_data.py <path_to_local_PROCESSED_VIDEOS_base_dir>")
        print("Example: python s3_upload_all_video_data.py ./processed-videos")
        sys.exit(1)

    local_dir = sys.argv[1]
    bucket = os.getenv("S3_BUCKET_NAME")

    upload_all_processed(local_dir, bucket)