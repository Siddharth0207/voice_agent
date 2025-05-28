from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import os
import sys 
import time 
import numpy as np
from uuid import UUID, uuid4
import requests
import json
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware # Import for handling CORS
from routers import summary, websocket_audio 
load_dotenv()
templates = Jinja2Templates(directory="templates")
app = FastAPI(
    title="Voice Agent API",
    description="Real-time speech-to-text + LangChain + NVIDIA endpoints",
    version="1.0.0",
)





app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allows all origins (for development purposes)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(summary.router, prefix="/api/v1/summary", tags=["Summary"])
app.include_router(websocket_audio.router, prefix="/api/v1/audio", tags=["WebSocket Audio"])

@app.get("/")
async def root():
    """
    Root endpoint for the FastAPI backend.

    Returns:
        dict: A welcome message indicating the API is running.
    """
    return ({"message": "this is Root"})