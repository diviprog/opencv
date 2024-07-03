import cv2
import pytesseract
import csv

# Read image using OpenCV
img_path = '/Users/devanshmishra/Desktop/Screenshot 2024-05-10 at 9.12.13 PM.png'
image = cv2.imread(img_path)

# Convert image to grayscale
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Apply thresholding or other preprocessing steps if needed
# For example, you can use adaptive thresholding:
# thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Perform OCR using Tesseract
# You can specify additional configuration options like language and page segmentation mode
text = pytesseract.image_to_string(image)
text = text.split('\n')
text = list(filter(None,text))
teams = ['Royal Challengers Bengaluru','Royal Challengers', 'Mumbai Indians', 'Gujarat Titans', 'Chennai Super Kings', 'Lucknow Super Giants', 'Delhi Capitals', 'Punjab Kings','Rajasthan Royals', 'Sunrisers Hyderabad', 'Kolkata Knight Riders']
for i in range(len(text)//3):
    for j in teams:
        if j in text[i]:
            text[i] = j
print(text)
# Parse the recognized text into rows and columns based on the table structure
# For a basic example, split the text into lines and then split each line into columns using a delimiter
lines = text.split('\n')
table_data = [line.split('\t') for line in lines]

# Write the table data to a CSV file
csv_path = '/Users/devanshmishra/Desktop/Top of the food chain/Devansh/Programming/cv_beginner/ipl_pointstable/table_data.csv'
with open(csv_path, 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(table_data)