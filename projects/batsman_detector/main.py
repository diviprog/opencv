'''
import cv2

# Load the pre-trained full body cascade
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Load the image
image = cv2.imread('/Users/devanshmishra/Downloads/vb07s9y88bs91.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect full bodies in the image
bodies = full_body_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around the detected bodies
for (x, y, w, h) in bodies:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

print(bodies)

# Display the result
cv2.imshow('Detected Bodies', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2

body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.1,3)

    for (x,y,w,h) in bodies:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0), 5)
        cv2.imshow('Detected Bodies', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()