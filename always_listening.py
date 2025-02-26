import queue
import sounddevice as sd
import vosk
import json
import sys

# Path to the Vosk model (update if your model is in a different directory)
model_path = "./vosk-model-small-en-us-0.15"

# Initialize the Vosk model
try:
    model = vosk.Model(model_path)
except Exception as e:
    sys.exit(f"Could not load the model at {model_path}: {e}")

# Set the sample rate; make sure this matches the model's requirements (typically 16000)
samplerate = 16000

# A thread-safe queue to hold audio data chunks
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """
    This callback is called for each audio block from the microphone.
    It places the raw audio data into a queue for the recognizer.
    """
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Open a raw input stream using sounddevice
with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16', channels=1, callback=audio_callback):
    # Initialize the recognizer with the model and sample rate
    recognizer = vosk.KaldiRecognizer(model, samplerate)
    print("Listening continuously... Press Ctrl+C to stop.")

    try:
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                # When a complete phrase is recognized, parse the final result.
                result = json.loads(recognizer.Result())
                print("Final Result:", result.get("text", ""))
            else:
                # Otherwise, print partial results.
                partial_result = json.loads(recognizer.PartialResult())
                print("Partial:", partial_result.get("partial", ""))
    except KeyboardInterrupt:
        print("\nStopping...")


