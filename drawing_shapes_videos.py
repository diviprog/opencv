import cv2

def draw_circle(video_path,centre_x,centre_y,radius):
    # Open a video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw a circle
        cv2.circle(frame, (centre_x, centre_y), radius, (255, 0, 0), 2)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_line(video_path,x1,y1,x2,y2):
    # Open a video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw a line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_rectangle(video_path,x1,x2,y1,y2):
    # Open a video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw a rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()