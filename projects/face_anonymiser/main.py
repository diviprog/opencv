import cv2
import mediapipe as mp

# blurring image
def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
                
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (30, 30))

    return img

# processing a static image
def static_image(img_path):
    img = cv2.imread(img_path)
    cv2.imshow('img', img_renderer(img, 1))
    cv2.waitKey(0)
        

# processing webcams
def web_cam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        frame = img_renderer(frame, 0)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def img_renderer(img, model_selection):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=0.5) as face_detection:
        return process_img(img, face_detection)


def main():
    #static_image('/Users/devanshmishra/Downloads/360_F_243123463_zTooub557xEWABDLk0jJklDyLSGl2jrr.jpg')
    web_cam()

# calls
main()
