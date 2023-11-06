# Load your eroded mask containing the suitable blobs
eroded_mask = cv2.imread('eroded_mask.png', 0)  # Load it in grayscale mode

# Initialize SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Set the threshold on the size of the blob you want to detect
params.filterByArea = True
params.minArea = 100  # Adjust this threshold based on the minimum size of the largest blob you want to detect

# Create a SimpleBlobDetector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the eroded mask
keypoints = detector.detect(eroded_mask)

# Find the largest blob by sorting the keypoints based on their size
keypoints = sorted(keypoints, key=lambda x: x.size, reverse=True)

# Extract the largest blob's coordinates
if keypoints:
    largest_blob = keypoints[0]

    # Get the mass center of the largest blob
    largest_blob_center = (int(largest_blob.pt[0]), int(largest_blob.pt[1]))

    # Draw the mass center on a copy of the original image
    largest_blob_image = cv2.cvtColor(eroded_mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR
    cv2.circle(largest_blob_image, largest_blob_center, 5, (0, 0, 255), -1)  # Draw a red circle at the mass center

    # Output the mass center coordinates to the terminal window
    print("Mass Center of the Largest Blob:", largest_blob_center)

    # Display or save the image with the mass center of the largest blob highlighted
    cv2.imshow('Blob extraction', largest_blob_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No blob found in the image")
