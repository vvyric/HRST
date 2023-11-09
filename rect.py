import cv2

# Load the template image
template_img = cv2.imread('templateImg.jpg')

# Manually specify the coordinates of the rectangle (x, y, width, height)
x, y, width, height = 100, 50, 200, 150

# Draw the rectangle on the template image
cv2.rectangle(template_img, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Extract the region of interest (ROI)
roi = template_img[y:y + height, x:x + width]

# Display the template image with the rectangle and the extracted ROI
cv2.imshow('Template Image with ROI', template_img)
cv2.imshow('Extracted ROI', roi)

# Save the extracted ROI as a separate image
cv2.imwrite('extracted_roi.jpg', roi)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
##############################


import pygame
import cv2
import numpy as np

# Load the template image
template_img = cv2.imread('templateImg.jpg')

# Create a pygame Rect object (x, y, width, height)
rect = pygame.Rect(100, 50, 200, 150)

# Draw the rectangle on the template image
cv2.rectangle(template_img, (rect.x, rect.y), (rect.x + rect.width, rect.y + rect.height), (0, 255, 0), 2)

# Extract the region of interest (ROI)
roi = template_img[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width]

# Display the template image with the rectangle and the extracted ROI
cv2.imshow('Template Image with ROI', template_img)
cv2.imshow('Extracted ROI', roi)

# Save the extracted ROI as a separate image
cv2.imwrite('extracted_roi.jpg', roi)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
