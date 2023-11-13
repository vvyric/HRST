
# Assuming you have a function to get images from NAO's camera
def get_image_from_nao_camera():
    # Replace with actual code to capture an image from NAO's camera
    return cv2.imread('test.jpg')  # Placeholder

# Initialize variables
old_frame = get_image_from_nao_camera()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Manually select the object to track or use object detection here
# For example, let's say you selected a region (x, y, w, h)
x, y, w, h = 100, 100, 50, 50  # Example coordinates
track_window = (x, y, w, h)

# Set the region of interest for tracking
roi = old_gray[y:y+h, x:x+w]

# Find feature points in the roi
p0 = cv2.goodFeaturesToTrack(roi, **feature_params)

# Adjust coordinates relative to the full image
p0[:,0,0] += x
p0[:,0,1] += y

# Create a mask for drawing the trajectory
mask = np.zeros_like(old_frame)

while True:
    frame = get_image_from_nao_camera()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select and draw good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('Object Tracking', img)

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
In this code, you'd replace the placeholder coordinates (x, y, w, h) with the actual coordinates of the object you want to track. You could get these coordinates by implementing a mouse callback function in OpenCV to select the object manually, or by using an object detection algorithm to find the object automatically. The rest of the code tracks the feature points within this selected region and draws their trajectory over time.




User
You
supppose I have already select ROI from a given image using cv2.selectROI, modify your code
ChatGPT
ChatGPT
If you've already selected a Region of Interest (ROI) from a given image using cv2.selectROI, the code can be modified to track the features within this ROI. The cv2.selectROI function allows you to interactively select a bounding box (ROI) on an image. After selecting the ROI, you can track the features within this region across frames. Here's how you can integrate it:

python
Copy code
import cv2
import numpy as np

# Assuming you have a function to get images from NAO's camera
def get_image_from_nao_camera():
    # Replace with actual code to capture an image from NAO's camera
    return cv2.imread('test.jpg')  # Placeholder

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize the first frame and select the ROI
first_frame = get_image_from_nao_camera()
roi_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Coordinates of the ROI
x, y, w, h = roi
roi_gray = roi_gray[y:y+h, x:x+w]

# Detect feature points in the ROI
p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

# Adjust coordinates relative to the full image
if p0 is not None:
    p0[:,0,0] += x
    p0[:,0,1] += y

# Create a mask for drawing the trajectory
mask = np.zeros_like(first_frame)

while True:
    frame = get_image_from_nao_camera()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None:
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, frame_gray, p0, None, **lk_params)

        # Select and draw good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('Object Tracking', img)

        # Update the previous frame and points for next iteration
        roi_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


#######################################3

import cv2
import numpy as np

# Function to get images from NAO's camera (You'll need to implement this)
def get_image_from_nao_camera():
    # Replace this with actual code to capture an image from the NAO robot's top camera
    return cv2.imread('test.jpg')  # Placeholder for an image

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
old_frame = get_image_from_nao_camera()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    frame = get_image_from_nao_camera()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
    img = cv2.add(frame, mask)

    # Display the image
    cv2.imshow('Frame', img)

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
