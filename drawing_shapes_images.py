import cv2

def draw_circle(img_path,centre,radius,color,thickness):
    # Read an image
    image = cv2.imread(img_path)

    # Draw a circle on the image
    cv2.circle(image, centre, radius, color, thickness)

    return image


def draw_line(img_path,start_point,end_point,color,thickness):
    # Read an image
    image = cv2.imread(img_path)

    # Draw a line on the image
    cv2.line(image, start_point, end_point, color, thickness)

    return image


def draw_rect(img_path,corner1,corner2,color,thickness):
    # Read an image
    image = cv2.imread(img_path)

    # Draw a rectangle on the image
    cv2.rectangle(image, corner1, corner2, color, thickness)

    return image


def add_text(img_path,text,position,color,thickness):
    # Read an image
    image = cv2.imread(img_path)

    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    return image