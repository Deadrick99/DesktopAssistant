import cv2
import os

if not os.path.exists("enrolled_faces"):
    os.makedirs("enrolled_faces")

cap = cv2.VideoCapture(0)
count = 0
while count < 5:
    ret, frame = cap.read()
    cv2.imshow('Enroll Your Face - Press "s" to save', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        cv2.imwrite(f"enrolled_faces/face_{count}.jpg", frame)
        print(f"Saved face_{count}.jpg")
        count += 1 
    elif key & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

