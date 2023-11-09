bin_width = hist_image.shape[1] // len(hist_normalized)
for i in range(len(hist_normalized)):
    x = i * bin_width
    y = int(hist_normalized[i])
    cv2.rectangle(hist_image, (x, hist_image.shape[0]), (x + bin_width, hist_image.shape[0] - y), (255, 255, 255), -1)



import cv2

# Load the image
image = cv2.imread('your_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Normalize the histogram to fit within the image height
hist_normalized = hist / hist.max() * 255

# Create a black image to draw the histogram
hist_image = np.zeros((256, 256, 3), dtype=np.uint8)

# Draw the histogram
for i in range(256):
    cv2.line(hist_image, (i, 256), (i, 256 - int(hist_normalized[i])), (255, 255, 255))

# Display the histogram
cv2.imshow('Histogram', hist_image)
cv2.waitKey(0)
cv2.destroyAllWindows()










import cv2
import numpy as np

# Load the incoming image
incoming_img = cv2.imread('incoming_image.jpg')

# Convert incoming image to HSV color space
incoming_hsv = cv2.cvtColor(incoming_img, cv2.COLOR_BGR2HSV)

# Split the channels
h, s, v = cv2.split(incoming_hsv)

# Mask out low-saturated pixels
saturation_threshold = 50
saturation_mask = cv2.threshold(s, saturation_threshold, 255, cv2.THRESH_BINARY)[1]

# Back project the hue channel using the normalized histogram
roi_hist_normalized = np.array(hist_normalized).reshape(-1, 1)
back_projection = cv2.calcBackProject([h], [0], roi_hist_normalized, [0, 256], scale=1)

# Bitwise AND the back projection result with the saturation mask
result = cv2.bitwise_and(back_projection, saturation_mask)

# Display the results
cv2.imshow('Incoming Image', incoming_img)
cv2.imshow('Back Projection', back_projection)
cv2.imshow('Saturation Mask', saturation_mask)
cv2.imshow('Result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
