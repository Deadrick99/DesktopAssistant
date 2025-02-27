import pvporcupine
import os
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

access_key = os.getenv('PORCUPINE_TOKEN')
print(access_key)
handle = pvporcupine.create(
    access_key=access_key,
    keyword_paths=['Jarvis_en_mac_v3_0_0/Jarvis_en_mac_v3_0_0.ppn'])

sample_rate = handle.sample_rate
frame_length = handle.frame_length

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio callback status:", status)
    # indata is a 2D NumPy array: (frames, channels). We use the first channel.
    pcm = indata[:, 0]
    
    # Process the current frame. If a wake word is detected, porcupine.process() returns a non-negative index.
    keyword_index = handle.process(pcm)
    if keyword_index >= 0:
        print("Wake word detected!")

try:
    # Open an input audio stream using sounddevice with the appropriate parameters.
    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=frame_length,
        dtype=np.int16,
        callback=audio_callback
    ):
        print("Listening for the wake word... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)  # Sleep in a loop to keep the stream active
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Clean up Porcupine resources
    handle.delete()
