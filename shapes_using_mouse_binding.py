import cv2

def circle_mouse_binding(img_path):
    # Mouse callback function
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 50, (255, 0, 0), -1)

    # Read an image
    image = cv2.imread(img_path)

    # Set up the window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()