import cv2
import numpy as np
import matplotlib.pyplot as plt

def eq_triangle(img_path):
    # Read an image
    image = cv2.imread('path_to_image.jpg')

    # Define triangle vertices
    pts = np.array([[150, 200], [100, 300], [200, 300]], np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw the triangle
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Calculate centroid
    centroid_x = int(np.mean(pts[:, 0, 0]))
    centroid_y = int(np.mean(pts[:, 0, 1]))

    # Draw the centroid
    cv2.circle(image, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

    # Display text
    text = centroid_x + ", " + centroid_y
    cv2.putText(image, text, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Equilateral Triangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_mouse_coordinates(img_path):
    # Mouse callback function
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Coordinates: ({x}, {y})")

    # Read an image
    image = cv2.imread(img_path)

    # Set up the window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()

def mouse_draw(img_path):
    drawing = False  # True if mouse is pressed
    ix, iy = -1, -1

    # Mouse callback function
    def draw(event, x, y, flags, param):
        global ix, iy, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(image, (x, y), 10, (0, 0, 0), -1)

    # Read an image
    image = cv2.imread(img_path)

    # Set up the window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()


def display_colour_of_pixel(img_path):
    # Mouse callback function
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            b, g, r = image[y, x]
            print(f"Pixel color at ({x}, {y}): R={r}, G={g}, B={b}")

    # Read an image
    image = cv2.imread('path_to_image.jpg')

    # Set up the window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()


def draw_box_using_mouse(img_path):
    # Initialize variables
    drawing = False
    ix, iy = -1, -1

    # Mouse callback function
    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing, image, clone_image
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                image = clone_image.copy()
                cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
            print(f"Rectangle coordinates: Top-left ({ix}, {iy}), Bottom-right ({x}, {y})")

    # Load an image
    image = cv2.imread(img_path)
    clone_image = image.copy()

    # Set up the window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()


def plot_hist_of_channels(img_path):
    # Load an image
    image = cv2.imread(img_path)

    # Compute the histogram for each channel
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    # Display the histogram
    plt.title('Histogram for color image')
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.show()


def bitwise_and(img_path1, img_path2):
    # Read two images of the same size
    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)

    # Perform bitwise AND operation
    and_image = cv2.bitwise_and(image1, image2)

    # Display the result
    cv2.imshow('AND Operation', and_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def object_classification(img_path):
    # Load the pre-trained model and labels
    net = cv2.dnn.readNetFromCaffe('model/deploy.prototxt', 'model/bvlc_googlenet.caffemodel')
    with open('model/synset_words.txt') as f:
        labels = f.read().strip().split("\n")

    # Read the image
    image = cv2.imread(img_path)

    # Prepare the image for classification
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
    net.setInput(blob)
    preds = net.forward()

    # Get the highest scoring class
    idx = np.argmax(preds[0])
    label = labels[idx]

    # Display the result
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def template_match(img_path, template_path):
    # Read the main image and the template image
    image = cv2.imread(img_path)
    template = cv2.imread(template_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    # Convert the main image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    res = cv2.matchTemplate(gray_image, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Draw a rectangle around the matched region
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Template Matching', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def feature_detection(img_path):
    # Read the image
    image = cv2.imread(img_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    # Draw the keypoints on the original image
    keypoints_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

    # Display the result
    cv2.imshow('ORB Feature Detection', keypoints_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def morphological_operations(img_path):
    # Read the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary threshold
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Define the kernel
    kernel = np.ones((5, 5), np.uint8)

    # Apply erosion
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Apply dilation
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Apply opening (erosion followed by dilation)
    opening_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Apply closing (dilation followed by erosion)
    closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Display the results
    cv2.imshow('Original', binary_image)
    cv2.imshow('Erosion', eroded_image)
    cv2.imshow('Dilation', dilated_image)
    cv2.imshow('Opening', opening_image)
    cv2.imshow('Closing', closing_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()