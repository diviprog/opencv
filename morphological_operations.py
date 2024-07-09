import cv2
import numpy as np

def dilate_image(image, size):
    # Ensure the image is in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Create the structuring element (kernel)
    kernel = np.ones((size, size), dtype=np.uint8)

    # Get the dimensions of the image
    height, width = binary_image.shape

    # Initialize the dilated image
    dilated_image = np.zeros_like(binary_image)

    # Define the bounds for kernel application
    lower_bound = (size - 1) // 2
    upper_bound = (size + 1) // 2

    # Perform dilation manually
    for i in range(lower_bound, height - lower_bound):
        for j in range(lower_bound, width - lower_bound):
            dilated_image[i, j] = np.max(binary_image[i - lower_bound:i + upper_bound, j - lower_bound:j + upper_bound]*kernel)

    return dilated_image


def erode_image(image,size):
    # Ensure the image is in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Create the structuring element (kernel)
    kernel = np.ones((size, size), dtype=np.uint8)

    # Get the dimensions of the image
    height, width = binary_image.shape

    # Initialize the eroded image
    eroded_image = np.zeros_like(binary_image)

    # Define the bounds for kernel application
    lower_bound = (size - 1) // 2
    upper_bound = (size + 1) // 2

    # Perform erosion manually
    for i in range(lower_bound, height - lower_bound):
        for j in range(lower_bound, width - lower_bound):
            eroded_image[i, j] = np.min(binary_image[i - lower_bound:i + upper_bound, j - lower_bound:j + upper_bound]*kernel)

    return eroded_image


def open_image(image,size):
    # Ensure the image is in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Create the structuring element (kernel)
    kernel = np.ones((size, size), dtype=np.uint8)

    # Get the dimensions of the image
    height, width = binary_image.shape

    # Initialize the dilated image
    dilated_image = np.zeros_like(binary_image)
    opened_image = np.zeros_like(binary_image)

    # Define the bounds for kernel application
    lower_bound = (size - 1) // 2
    upper_bound = (size + 1) // 2

    # Perform erosion manually
    for i in range(lower_bound, height - lower_bound):
        for j in range(lower_bound, width - lower_bound):
            opened_image[i, j] = np.min(binary_image[i - lower_bound:i + upper_bound, j - lower_bound:j + upper_bound]*kernel)
    
    # Perform dilation manually
    for i in range(lower_bound, height - lower_bound):
        for j in range(lower_bound, width - lower_bound):
            dilated_image[i, j] = np.max(binary_image[i - lower_bound:i + upper_bound, j - lower_bound:j + upper_bound]*kernel)    

    return opened_image


def close_image(image,size):
    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Create the structuring element (kernel)
    kernel = np.ones((size, size), dtype=np.uint8)

    # Get the dimensions of the image
    height, width = binary_image.shape

    # Initialize the dilated image
    dilated_image = np.zeros_like(binary_image)
    closed_image = np.zeros_like(binary_image)

    # Define the bounds for kernel application
    lower_bound = (size - 1) // 2
    upper_bound = (size + 1) // 2

    # Perform dilation manually
    for i in range(lower_bound, height - lower_bound):
        for j in range(lower_bound, width - lower_bound):
            dilated_image[i, j] = np.max(binary_image[i - lower_bound:i + upper_bound, j - lower_bound:j + upper_bound]*kernel)

    # Perform erosion manually
    for i in range(lower_bound, height - lower_bound):
        for j in range(lower_bound, width - lower_bound):
            closed_image[i, j] = np.min(binary_image[i - lower_bound:i + upper_bound, j - lower_bound:j + upper_bound]*kernel)  

    return closed_image


def add(image1,image2):
    return cv2.add(image1,image2)


def subtract(image1,image2):
    return cv2.subtract(image1,image2)


def morphological_gradient(image):
    return subtract(dilate_image(image,5),erode_image(image,5))


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