import cv2
import numpy as np

# Define the lower and upper boundaries of the red color in HSV color space
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Define a 5x5 kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Start capturing the video
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply color filter to detect the red color in the frame
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Apply morphological operations to remove noise and fill gaps in the detected object
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours and draw bounding boxes around the detected objects
    for cnt in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Draw a rectangle around the contour if it meets the size requirements
        if w > 20 and h > 20 and w*h > 400:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

