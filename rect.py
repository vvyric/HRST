import cv2 
import numpy as np 
  
  
# Read image 
image = cv2.imread("image.png") 
  
# Select ROI 
r = cv2.selectROI("select the area", image) 
  
# Crop image 
cropped_image = image[int(r[1]):int(r[1]+r[3]),  
                      int(r[0]):int(r[0]+r[2])] 
  
# Display cropped image 
cv2.imshow("Cropped image", cropped_image) 
cv2.waitKey(0)









import cv2
import numpy as np

# Load the template image
template_img = cv2.imread('templateImg.jpg')

# Define the region of interest (ROI)
x, y, width, height = 100, 50, 200, 150
roi = template_img[y:y + height, x:x + width]

# Convert ROI to HSV color space
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Split the channels
h, s, v = cv2.split(roi_hsv)

# Threshold to mask out low-saturated pixels
saturation_threshold = 50
_, mask = cv2.threshold(s, saturation_threshold, 255, cv2.THRESH_BINARY)

# Compute the 1D histogram
hist = cv2.calcHist([roi_hsv], [0], mask, [256], [0, 256])

# Normalize the histogram
hist_normalized = hist / hist.sum()

# Display the results
cv2.imshow('ROI', roi)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
