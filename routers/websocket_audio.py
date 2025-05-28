from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import json
import numpy as np
import asyncio
from fastapi import APIRouter

router = APIRouter()

# Load the Whisper model globally when the app starts
# You can change the model size and device
model = WhisperModel("base", device="cpu", compute_type="int8")
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHUNK_DURATION_SEC = 2
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH


@router.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription using WhisperModel.

    Receives audio data in chunks from the client, buffers it, and processes it in segments.
    Performs speech-to-text transcription on each segment using the Whisper model and sends
    partial and final transcription results back to the client in real time.

    Args:
        websocket (WebSocket): The WebSocket connection instance.

    Protocol:
        - Receives binary audio data (PCM 16-bit, 16kHz) from the client.
        - Sends JSON messages with partial and final transcriptions.
        - Handles control messages (e.g., to indicate end of utterance).
    """
    await websocket.accept()  # Accept the WebSocket connection
    partial_buffer = bytearray()  # Buffer for partial audio chunks
    final_buffer = bytearray()    # Buffer for the full utterance
    try:
        while True:
            data = await websocket.receive()  # Receive data from the client

            if "bytes" in data:
                chunk = data["bytes"]  # Audio chunk from client
                partial_buffer += chunk  # Add to partial buffer
                final_buffer += chunk    # Add to final buffer

                # If enough audio is buffered, process it
                if len(partial_buffer) >= CHUNK_BYTES:
                    to_process = partial_buffer[:CHUNK_BYTES]
                    partial_buffer = partial_buffer[CHUNK_BYTES:]

                    # Convert bytes to float32 numpy array for Whisper
                    audio_array = np.frombuffer(to_process, dtype=np.int16).astype(np.float32) / 32768.0
                    # Transcribe the audio chunk (partial)
                    segments, _ = model.transcribe(audio_array, beam_size=5, language="en", vad_filter=True)
                    # Combine all segment texts
                    text = " ".join([s.text for s in segments])
                    # Send partial transcription to client
                    await websocket.send_text(json.dumps({"type": "partial", "text": text}))

            elif "text" in data:
                control = json.loads(data["text"])  # Control message from client
                if control.get("final"):
                    # If client signals end of utterance, process the full buffer
                    audio_array = np.frombuffer(final_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = model.transcribe(audio_array, beam_size=5, language="en", vad_filter=True)
                    full_text = " ".join([s.text for s in segments])
                    # Send final transcription to client
                    await websocket.send_text(json.dumps({"type": "final", "text": full_text}))
                    # Reset buffers for next utterance
                    partial_buffer = bytearray()
                    final_buffer = bytearray()

            await asyncio.sleep(0.001)  # Yield to event loop

    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")  # Log any errors
    finally:
        await websocket.close()  # Ensure the WebSocket is closed