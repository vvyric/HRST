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
