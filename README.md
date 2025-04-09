# @AskPerplexity AI Agent For Video Q&A (RAG+MCP PoC)

@AskPerplexity feature on X, but for TikTok For You page #fyp

inspired by Perplexity's recent blog post "A Vision for Rebuilding TikTok in America":
https://www.perplexity.ai/hub/blog/rebuilding-tiktok-in-america

## Description

This project demonstrates a TikTok-style video feed application where users can ask questions about the video content and receive AI-generated answers asynchronously.

## Core Features

- **Vertical Video Feed:** Displays a scrollable feed of short-form videos.
- **Asynchronous Q&A:** Ask questions via comment that tag @AskPerplexity; user does not wait, keeps scrolling feed as normal, and gets notified when the AI answer is ready.
- **RAG + MCP Pipeline:** Uses Retrieval-Augmented Generation (RAG) with Pinecone to augment user queries with relevant context from the video. Also leverages MCP to query Perplexity's Model Context Protocol (MCP) server for enhanced, context-aware answers.

## UX Flow Summary

1.  **Launch:** User opens the app and sees the "For You" feed with a video playing.
2.  **Browse:** User swipes/scrolls vertically through the initial videos.
3.  **Ask:** User taps comment icon, types question, submits. The question appears immediately (optimistic UI). The app sends the request to the backend and the user continues browsing.
4.  **Ask (New Video - Optional):** User navigates to "Process New Video", submits a TikTok URL + question. Processing starts in the background.
5.  **Get Notified:** After background AI processing completes (seconds to minutes), a UI notification indicates an answer is ready.
6.  **View:** User views the AI-generated answer as a reply in the comment section for the relevant video.

## Architecture Overview

- **Frontend (Next.js on Vercel):** Handles UI, user interaction, calls Backend API.
- **Backend (FastAPI on AWS App Runner):** Single API service in a Docker container. Handles requests, runs AI pipeline via background tasks.
- **Storage (AWS S3):** Stores video files (`.mp4`) and JSON metadata files (captions, summaries, Q&A state/answers).
- **Vector DB (Pinecone):** Stores video caption embeddings for fast retrieval.
- **AI Services:** Gemini (Captions), OpenAI (Embeddings, Synthesis), Perplexity (MCP - Reasoning/Search).

## Backend (FastAPI / Python)

- **Technology:** FastAPI, Uvicorn, Docker, Boto3 (AWS SDK).
- **Core Logic (`app/pipeline_logic.py`):** Contains functions for downloading, chunking, captioning (Gemini), summarizing (OpenAI), indexing (OpenAI Embeddings -> Pinecone), retrieving context (Pinecone), calling MCP (Perplexity), synthesizing answers (OpenAI).
- **API Endpoints (`app/main.py`):**
  - `GET /api/videos/foryou`: Provides initial pre-processed video URLs (from S3).
  - `POST /api/query/async`: Triggers background task (query pipeline) for a pre-processed video Q&A. Returns immediately.
  - `POST /api/process_and_query/async`: Triggers background task (full pipeline: download -> ... -> answer) for a new video URL + Q&A. Returns immediately.
  - `GET /api/query/status/{video_id}`: Frontend polls this to get Q&A status and answers (reads from S3 JSON).
- **State Management:** Q&A status (`processing`, `completed`, `failed`) and AI answers are stored within an `interactions` array inside the corresponding video's JSON metadata file on S3. Background tasks update this file.
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
2.  **Pre-process Videos:** Manually run scripts to process 10 videos: Upload `.mp4` and `.json` (with empty `interactions`) to `s3://<your-bucket>/preprocessed/`, upload embeddings to Pinecone.
3.  **Backend:**
    - `cd backend`
    - Create/populate `.env` with secrets for local testing.
    - `pip install -r requirements.txt`
    - Local Test: `uvicorn app.main:app --reload`
    - Deploy: Build/push Docker image to ECR, deploy via AWS App Runner (configure Env Vars & IAM Role).
4.  **Frontend:**
    - `cd frontend`
    - Create `.env.local`, add `NEXT_PUBLIC_API_BASE_URL=<your_deployed_backend_url>`.
    - `npm install`
    - Local Test: `npm run dev` (ensure backend is running/deployed).
    - Deploy: Connect repo to Vercel, set Root Directory to `frontend`, configure `NEXT_PUBLIC_API_BASE_URL`.
5.  **CORS:** Configure CORS on AWS S3 bucket (allow GET from frontend domain) and in backend FastAPI `main.py` (allow requests from frontend domain).

## Key Trade-offs to prioritize speed of development (to be improved in future)

- **Polling:** Frontend uses basic polling (`setInterval`) for status updates, less efficient than WebSockets.
- **S3 JSON State:** Storing Q&A state in S3 JSON involves inefficient read-modify-write operations; DynamoDB would be better at scale.
- **Background Tasks:** FastAPI's `BackgroundTasks` are simple but not persistent; server restarts lose running tasks. Celery/Redis would be more robust.
