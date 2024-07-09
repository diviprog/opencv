import cv2

def img_flip(img_path,direction):
    # Read an image
    image = cv2.imread(img_path)

    if direction == "vertical" or direction == "Vertical":
        num = 0
    elif direction == "horizontal" or direction == "Horizontal":
        num = 1
    else:
        num = -1

    # Flip the image
    flipped_image = cv2.flip(image, num)  # 0: vertical flip, 1: horizontal flip, -1: both

    return flipped_image


def img_rotation(img_path,deg):
    # Read an image
    image = cv2.imread(img_path)

    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # Define the center of the image
    center = (w // 2, h // 2)

    # Rotate the image by 45 degrees
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    return rotated_image


def img_resize(img_path, new_size):
    # Read an image
    image = cv2.imread(img_path)

    # Resize the image
    resized_image = cv2.resize(image, new_size)  # New size (width, height)

    return resized_image


def img_blur(img_path,kernel):
    # Read an image
    image = cv2.imread(img_path)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (kernel, kernel), 0)

    return blurred_image


def img_splitting(img_path):
    # Read an image
    image = cv2.imread(img_path)

    # Split the image into its B, G, and R channels
    B, G, R = cv2.split(image)

    return B, G, R


def img_cannyedge(img_path):
    # Read an image
    image = cv2.imread(img_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)  # Threshold values

    return edges