import cv2
import datetime

def date_time(video_path):

    # Open a video file
    cap = cv2.VideoCapture('path_to_video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get current date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")

        # Add date and time to frame
        cv2.putText(frame, date_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Date and Time on Video', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
