import cv2
import os
import face_recognition
import pickle
import pyttsx3
import numpy
import requests  
import subprocess
from dotenv import load_dotenv

load_dotenv()

engine = pyttsx3.init()

with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

def speak (text, voice ="Jamie (Premium)"):
    subprocess.run(["say", "-v", voice, text])

def get_weather():
    API_KEY = os.getenv("API_TOKEN")
    url = f"http://api.openweathermap.org/data/2.5/weather?zip=37604,US&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"{weather} with a temperature of {temp} F"
    else:
        return "weather data unavailible"

def get_calendar_events():
    return "You have a meeting at 10 am and a lunch appointment at 1 PM."

def authenticate_face():
    authenticated = False
    video_capture = cv2.VideoCapture(0)
    while not authenticated:
        ret, frame = video_capture.read()
        rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=0)

        for(top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_enc)
            name = "Unknown"
            if True in matches:
                matched_idx = [i for (i, b) in enumerate(matches) if b]
                name = known_names[matched_idx[0]]

            cv2.rectangle(frame, (left,top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name != "Unknown":
                authenticated = True
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
    video_capture.release()
    cv2.destroyAllWindows()
    return


while True:
    authenticate_face()
    weather_info = get_weather()
    calendar_info = get_calendar_events()
    greeting_text = f"Good morning sir! Today's weather is {weather_info}. {calendar_info}"
    speak(greeting_text)
    print(greeting_text)
