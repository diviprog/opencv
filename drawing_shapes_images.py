import cv2

def draw_circle(img_path,centre_x,centre_y,radius):
    # Read an image
    image = cv2.imread(img_path)

    # Draw a circle on the image
    centre_coordinates = (centre_x, centre_y)
    color = (0, 255, 0)
    thickness = 2
    cv2.circle(image, centre_coordinates, radius, color, thickness)

    # Display the image
    cv2.imshow('Image with Circle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_line(img_path,start_point,end_point):
    # Read an image
    image = cv2.imread(img_path)

    # Draw a line on the image
    color = (255, 0, 0) 
    thickness = 2
    cv2.line(image, start_point, end_point, color, thickness)

    # Display the image
    cv2.imshow('Image with Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_rect(img_path,x1,x2,y1,y2,color,thickness):
    # Read an image
    image = cv2.imread(img_path)

    # Draw a rectangle on the image
    top_left_corner = (x1, y1)
    bottom_right_corner = (x2, y2) 
    cv2.rectangle(image, top_left_corner, bottom_right_corner, color, thickness)

    # Display the image
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_text(img_path,text,x1,y1):
    # Read an image
    image = cv2.imread(img_path)

    # Add text to the image
    position = (x1,y1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2  # Thickness of the text
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Image with Text', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()