# @AskPerplexity AI Agent For Video Q&A (RAG+MCP PoC)

@AskPerplexity feature on X, but for TikTok For You page #fyp

inspired by Perplexity's recent blog post "A Vision for Rebuilding TikTok in America":
https://www.perplexity.ai/hub/blog/rebuilding-tiktok-in-america

## Description

This project demonstrates a TikTok-style video feed application where users can ask questions about the video content and receive AI-generated answers asynchronously. Videos are processed and stored, allowing users to query them.

## Core Features

- **Vertical Video Feed:** Displays a scrollable feed of videos whose data is already fully processed and stored (`processing_status` is `FINISHED`).
- **Asynchronous Q&A:** Ask questions via comment that tag @AskPerplexity; user does not wait, keeps scrolling feed as normal, and gets notified when the AI answer is ready.
- **RAG + MCP Pipeline:** Uses Retrieval-Augmented Generation (RAG) with Pinecone to augment user queries with relevant context from the video. Also leverages MCP to query Perplexity's Model Context Protocol (MCP) server for enhanced, context-aware answers.

## UX Flow Summary

1.  **Launch:** User opens the app and sees the "For You" feed with a video playing (sourced from videos marked as `FINISHED` in S3).
2.  **Browse:** User swipes/scrolls vertically through the videos.
3.  **Ask (Processed Video):** User taps comment icon on a video from the feed, types question, submits. The question appears immediately (optimistic UI). The app sends the request to the backend (`/api/query/async`) and the user continues browsing.
4.  **Ask (New Video):** User navigates to "Process New Video", submits a TikTok URL + question. The backend (`/api/process_and_query/async`) creates metadata JSON with `processing_status: PROCESSING`, processes the video fully (download, analyze, store), handles the query, and finally updates `processing_status` to `FINISHED`. Processing happens in the background.
5.  **Get Notified:** After background AI processing completes (seconds to minutes), a UI notification indicates an answer is ready.
6.  **View:** User views the AI-generated answer as a reply in the comment section for the relevant video.

## Architecture Overview

- **Frontend (Next.js on Vercel):** Handles UI, user interaction, calls Backend API.
- **Backend (FastAPI on AWS App Runner):** Single API service in a Docker container. Handles requests, runs AI pipeline via background tasks.
- **Storage (AWS S3):** Stores video data under a `video-data/` prefix. Each processed video gets a dedicated folder `video-data/<video_id>/`. This folder contains:
  - `<video_id>.json`: Metadata file holding video details (e.g., captions, summary, themes) and the overall `processing_status` (`PROCESSING` or `FINISHED`). This file is separate from interaction data to reduce concurrency issues.
  - `interactions.json`: A separate file containing an array of user Q&A interactions. Created upon the first query to a video, and updated with each subsequent query. Each interaction includes the query, status (`processing`, `completed`, `failed`), AI answer, timestamps, and `interaction_id`. Keeping interactions separate from video metadata prevents conflicts when processing multiple queries simultaneously.
  - **S3 Bucket Policy:** The bucket is configured with a policy that grants public read access to all MP4 files in the `video-data/` prefix. This allows direct use of S3 object URLs for video display without requiring pre-signed URLs or authentication.
- **Video ID Format:** Each video is uniquely identified by a `video_id` that combines the TikTok username and video ID from the URL, separated by a dash. For example, from the URL `https://www.tiktok.com/@aichifan33/video/7486040114695507242`, the `video_id` would be `aichifan33-7486040114695507242`.
- **Vector DB (Pinecone):** Stores video caption embeddings for fast retrieval.
- **AI Services:** Gemini (Captions), OpenAI (Embeddings, Synthesis), Perplexity (MCP - Reasoning/Search).

## Backend (FastAPI / Python)

- **Technology:** FastAPI, Uvicorn, Docker, Boto3 (AWS SDK).
- **Core Logic (`app/pipeline_logic.py`):** Contains functions for downloading, chunking, captioning (Gemini), summarizing (OpenAI), indexing (OpenAI Embeddings -> Pinecone), retrieving context (Pinecone), calling MCP (Perplexity), synthesizing answers (OpenAI).
- **API Endpoints (`app/main.py`):**
  - `GET /api/videos/foryou`: Provides video URLs for videos whose `<video_id>.json` file in S3 has `processing_status: FINISHED`.
  - `POST /api/query/async`: Triggers a background task for an existing video (status `FINISHED`). The task adds/updates the query in `interactions.json` (creating the file if needed) and sets its status to `processing`. Returns immediately.
  - `POST /api/process_and_query/async`: Creates the initial `<video_id>.json` with `processing_status: PROCESSING`. Triggers a background task for the full pipeline. This task also creates/updates `interactions.json` with the initial user query (status `processing`). After indexing, it updates `processing_status` to `FINISHED` in `<video_id>.json`. Finally, it updates the specific interaction's status in `interactions.json` upon completion/failure. Stores results under `video-data/<video_id>/`. Returns immediately.
  - `GET /api/query/status/{video_id}`: Frontend polls this. Reads `<video_id>.json` for `processing_status` and `interactions.json` for the list of Q&A interactions.
- **State Management:** Overall processing state (`processing_status`) is in `<video_id>.json`. Individual Q&A interaction states are managed within the array in `interactions.json`. Both are stored in S3 under `video-data/<video_id>/`. Background tasks update these files.
- **Deployment:** Docker image built from `backend/Dockerfile`, pushed to AWS ECR, deployed via AWS App Runner (configured with environment variables for API keys/secrets and an IAM Role for S3 access). Requires CORS configuration to allow requests from the frontend domain.

## Frontend (Next.js / TypeScript)

- **Technology:** Next.js 14+ (App Router), TypeScript, Tailwind CSS.
- **Core Logic (`src/`):**
  - `app/page.tsx` fetches initial videos (Server Component).
  - `components/FeedClient.tsx` (`'use client'`) manages feed display and scrolling.
  - `components/CommentSection.tsx` (`'use client'`) handles question input, optimistic UI updates, triggers `POST /api/query/async`, displays fetched Q&A interactions.
  - A global polling mechanism (`'use client'`, e.g., in Context or `NotificationManager.tsx`) periodically calls `GET /api/query/status/{video_id}` using `setInterval` and updates shared state when answers are ready.
- **State Management:** Primarily React Hooks (`useState`, `useContext`, `useReducer`) for managing feed state, comment visibility, and Q&A interaction status/answers.
- **Deployment:** Deployed via Vercel by connecting the Git repository (monorepo). Root Directory set to `frontend`. Requires `NEXT_PUBLIC_API_BASE_URL` environment variable pointing to the deployed backend URL.

## Setup & Running (Quick Start)

1.  **Prerequisites:** Ensure accounts/API keys for AWS, Pinecone, OpenAI, Google Cloud (Gemini), Perplexity. Install Node.js, Python, Docker, Git, AWS CLI.
2.  **Process Initial Videos:** Manually run scripts or use the process endpoint. Ensure for each video, `<video_id>.json` exists in `s3://<your-bucket>/video-data/<video_id>/` with `processing_status: FINISHED`. The `interactions.json` file should _not_ exist initially; it will be created upon the first query. Ensure chunk captions are indexed into Pinecone.
3.  **S3 Configuration:**
    - Create an S3 bucket and configure a bucket policy that allows public read access to all objects with the `.mp4` extension under the `video-data/` prefix.
    - Example bucket policy snippet:
      ```json
      {
        "Version": "2012-10-17",
        "Statement": [
          {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::<your-bucket-name>/video-data/*/*.mp4"
          }
        ]
      }
      ```
    - Configure CORS on your S3 bucket to allow direct video loading from your frontend domain:
      ```json
      [
        {
          "AllowedHeaders": ["*"],
          "AllowedMethods": ["GET"],
          "AllowedOrigins": ["https://your-frontend-domain.com"],
          "ExposeHeaders": []
        }
      ]
      ```
4.  **Backend:**
    - `cd backend`
    - Create/populate `.env` with secrets for local testing.
    - `pip install -r requirements.txt`
    - Local Test: `uvicorn app.main:app --reload`
    - Deploy: Build/push Docker image to ECR, deploy via AWS App Runner (configure Env Vars & IAM Role).
    - **CORS Configuration:** Update the `origins` list in `app/main.py` to include your frontend domain:
      ```python
      origins = [
          "http://localhost:3000",  # For local development
          "https://your-frontend-domain.com",  # Your deployed frontend URL
      ]
      ```
5.  **Frontend:**
    - `cd frontend`
    - Create `.env.local`, add `NEXT_PUBLIC_API_BASE_URL=<your_deployed_backend_url>`.
    - `npm install`
    - Local Test: `npm run dev` (ensure backend is running/deployed).
    - Deploy: Connect repo to Vercel, set Root Directory to `frontend`, configure `NEXT_PUBLIC_API_BASE_URL`.

## CORS Configuration

This application requires CORS configuration in two places due to its architecture:

1. **Backend API CORS (FastAPI):** Because your frontend and backend run on different domains (e.g., `your-app.vercel.app` and `xxx.awsapprunner.com`), the browser would normally block API requests from the frontend to the backend due to the Same-Origin Policy. The CORS middleware in FastAPI needs to be configured to explicitly allow requests from your frontend domain for these API endpoints:

   - `GET /api/videos/foryou` - For fetching the video feed
   - `POST /api/query/async` - For submitting queries to preprocessed videos
   - `POST /api/process_and_query/async` - For processing new videos and queries
   - `GET /api/query/status/{video_id}` - For polling query status

2. **S3 CORS:** Since the frontend directly loads videos from S3 URLs, the S3 bucket needs CORS configuration to allow requests from your frontend domain.

Without proper CORS configuration, your application will fail with errors like "Access to fetch at 'https://xxx.awsapprunner.com/api/videos/foryou' from origin 'https://your-app.vercel.app' has been blocked by CORS policy."

## Key Trade-offs to prioritize speed of development (to be improved in future)

- **Polling:** Frontend uses basic polling (`setInterval`) for status updates, less efficient than WebSockets.
- **S3 JSON State:** Storing state in S3 JSON involves read-modify-write. Separating interactions helps, but DynamoDB would still be more robust for frequent interaction updates.
- **Background Tasks:** FastAPI's `BackgroundTasks`
