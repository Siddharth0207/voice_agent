import sounddevice as sd
import numpy as np
from io import BytesIO
import time
import asyncio
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor



executor = ThreadPoolExecutor()

def _record_until_silence_blocking(
    fs=16000, 
    max_duration=180,
    min_silence_len_ms=2000,
    silence_thresh_db=-35,
    device=None
    ):

    """
    Records audio until silence is detected or the maximum duration is reached.

    Args:
        fs (int): Sampling rate in Hz. Default is 16000.
        max_duration (int): Maximum recording duration in seconds. Default is 180 seconds.
        min_silence_len_ms (int): Minimum silence duration in milliseconds to stop recording. Default is 1000 ms.
        silence_thresh_db (int): Silence threshold in dBFS. Default is -30 dB.
        device (int or None): Audio input device ID. Default is None (uses the default device).

    Returns:
        tuple: A tuple containing the recorded audio as a NumPy array and the sampling rate (int).
    """

    audio_buffer = []
    start_time = time.time()


    def callback(indata, frames, time_info, status):
        """
        Callback function for the audio stream to process incoming audio data.

        Args:
            indata (numpy.ndarray): The audio input data.
            frames (int): The number of frames in the input data.
            time_info (dict): Timing information for the audio stream.
            status (sounddevice.CallbackFlags): Status flags for the audio stream.
        """
        if status:
            print(f"[Stream Status] {status}")
        # Flatten the mono channel
        audio_buffer.extend(indata[:, 0].copy())

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback, device=device):
        #Loop untill the maximum duration is reached
        while time.time() - start_time < max_duration:
            time.sleep(0.5)# Sleep for 0.5 seconds to allow audio to accumulate

            if len(audio_buffer) == 0:
                # Skips the loop or Processing if no audio is recorded yet
                continue
            # Convert the audio buffer to a NumPy array for processing
            np_audio = np.array(audio_buffer, dtype='int16')

            # Calculate the number of samples corresponding to the minimum silence length
            tail_samples = int(fs * min_silence_len_ms / 1000)
            if len(np_audio) < tail_samples:
                # Skip if there isn't enough audio data to analyze the last portion
                continue

            # Extract the last portion of audio corresponding to the minimum silence length
            np_tail = np_audio[-tail_samples:]

            # Write the extracted portion to a BytesIO buffer as a valid WAV file
            buffer = BytesIO()
            wav_write(buffer, fs, np_tail)  # Write the audio data to the buffer
            buffer.seek(0)  # Reset the buffer's position to the beginning

            try:
                # Load the audio segment from the buffer for silence analysis
                tail_seg = AudioSegment.from_file(buffer, format="wav")
            except Exception as e:
                # Handle errors that occur while loading the audio segment
                print(f"[Error loading audio segment]: {e}")
                continue

            # Log dBFS for tuning
            # Analyze the dBFS (decibels relative to full scale) of the audio segment
            end_slice = tail_seg
            # Uncomment the line below to debug the dBFS value of the audio segment
            #print(f"[DEBUG] dBFS: {end_slice.dBFS:.2f}")

            # Check if the dBFS value is below the silence threshold
            if end_slice.dBFS < silence_thresh_db:
                print("🔇 Detected silence. Ending recording.")
                # Return the recorded audio and the sampling rate
                return np_audio, fs
    # If the maximum duration is reached, return the recorded audio and sampling rate
    return np.array(audio_buffer, dtype='int16'), fs


# Async wrapper for FastAPI use
async def record_until_silence_async(
    fs=16000,
    max_duration=180,
    min_silence_len_ms=2000,
    silence_thresh_db=-35,
    user_id=None
):
    """
    Asynchronous wrapper for recording audio until silence is detected or the maximum duration is reached.

    Args:
        fs (int): Sampling rate in Hz. Default is 16000.
        max_duration (int): Maximum recording duration in seconds. Default is 180 seconds.
        min_silence_len_ms (int): Minimum silence duration in milliseconds to stop recording. Default is 1000 ms.
        silence_thresh_db (int): Silence threshold in dBFS. Default is -40 dB.
        user_id (str or None): Optional user ID for tracking purposes. Default is None.

    Returns:
        tuple: A tuple containing the recorded audio as a NumPy array and the sampling rate (int).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        _record_until_silence_blocking,
        fs,
        max_duration,
        min_silence_len_ms,
        silence_thresh_db
    )
