import cv2
import numpy as np
import matplotlib.pyplot as plt
from drawing_shapes_images import add_text

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
        if cv2.waitKey(0):
            break

    cv2.destroyAllWindows()

def mouse_draw(img_path):
    drawing = False
    ix, iy = -1, -1

    # Mouse callback function
    def draw(event, x, y, flags, param):
        global drawing,ix,iy
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
        if cv2.waitKey(0):
            break

    cv2.destroyAllWindows()


def display_colour_of_pixel(img_path):
    # Mouse callback function
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            b, g, r = image[y, x]
            print(f"Pixel color at ({x}, {y}): R={r}, G={g}, B={b}")

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

    return and_image


def bitwise_or(img_path1, img_path2):
    # Read two images of the same size
    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)

    # Perform bitwise AND operation
    or_image = cv2.bitwise_or(image1, image2)

    return or_image

def bitwise_not(img_path1):
    # Read two images of the same size
    image1 = cv2.imread(img_path1)

    # Perform bitwise AND operation
    not_image = cv2.bitwise_not(image1)

    return not_image


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
