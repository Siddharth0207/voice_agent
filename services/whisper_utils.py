from faster_whisper import WhisperModel
import tempfile
import scipy.io.wavfile as wav
import numpy as np
import asyncio

async def load_whisper_model(size="tiny", device="cpu"):
    """
    Loads the Whisper model with the specified size and device.

    Args:
        size (str): The size of the Whisper model to load. Default is "tiny".
        device (str): The device to load the model on ("cpu" or "cuda"). Default is "cpu".

    Returns:
        WhisperModel: An instance of the Whisper model loaded with the specified configuration.
    """
    
    compute = "int8" if device == "cpu" else "float16"
    return WhisperModel(size, device=device, compute_type=compute)

async def transcribe_audio(model, audio_data, sample_rate=16000):
    """     Transcribes audio data using the provided Whisper model.

    Args:
        model (WhisperModel): The Whisper model instance to use for transcription.
        audio_data (numpy.ndarray): The audio data to transcribe, represented as a NumPy array.
        sample_rate (int): The sampling rate of the audio data in Hz. Default is 16000.

    Returns:
        str: The transcription result as a single string. """   
    # Save to temp WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        wav.write(temp_wav.name, sample_rate, np.array(audio_data, dtype='int16'))

        segments, _ =  model.transcribe(temp_wav.name)
        transcription_result = " ".join([seg.text for seg in segments])
        return transcription_result