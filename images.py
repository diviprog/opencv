import cv2

def read_an_image(img_path):
    return cv2.imread(img_path)

def read_and_display_an_image(img_path):
    # Read an image
    image = cv2.imread(img_path)

    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_an_image(image):
    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_an_image(file_path, image):
    cv2.imwrite(file_path, image)