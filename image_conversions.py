import cv2

def read_in_grayscale(img_path):
    # Read an image in grayscale
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Display the grayscale image
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def convert_coloured_to_grayscale(img_path):
    # Read a colored image
    image = cv2.imread(img_path)

    # Convert the colored image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def convert_grayscale_to_binary(img_path):
    # Read a grayscale image
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Convert the grayscale image to binary using a threshold
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Display the binary image
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def convert_bgr_to_rgb(img_path):
    # Read a colored image
    image = cv2.imread(img_path)

    # Convert the image from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the RGB image
    cv2.imshow('RGB Image', rgb_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()