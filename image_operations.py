import cv2

def img_flip(img_path):
    # Read an image
    image = cv2.imread(img_path)

    # Flip the image
    flipped_image = cv2.flip(image, 1)  # 0: vertical flip, 1: horizontal flip, -1: both

    # Display the flipped image
    cv2.imshow('Flipped Image', flipped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_rotation(img_path):
    # Read an image
    image = cv2.imread(img_path)

    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # Define the center of the image
    center = (w // 2, h // 2)

    # Rotate the image by 45 degrees
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Display the rotated image
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_resize(img_path):
    # Read an image
    image = cv2.imread(img_path)

    # Resize the image
    resized_image = cv2.resize(image, (300, 300))  # New size (width, height)

    # Display the resized image
    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_blur(img_path):
    # Read an image
    image = cv2.imread('path_to_image.jpg')

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)  # Kernel size (15, 15)

    # Display the blurred image
    cv2.imshow('Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_splitting(img_path):
    # Read an image
    image = cv2.imread('path_to_image.jpg')

    # Split the image into its B, G, and R channels
    (B, G, R) = cv2.split(image)

    # Display each channel
    cv2.imshow('Blue Channel', B)
    cv2.imshow('Green Channel', G)
    cv2.imshow('Red Channel', R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_cannyedge(img_path):
    # Read an image
    image = cv2.imread('path_to_image.jpg')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)  # Threshold values

    # Display the edges
    cv2.imshow('Canny Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()