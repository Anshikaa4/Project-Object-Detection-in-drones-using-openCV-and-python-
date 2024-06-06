import cv2
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to the input video")
args = vars(ap.parse_args())

# Access the video path from the parsed arguments
video_path = args["video"]

# Open the video file
camera = cv2.VideoCapture(video_path)

# Loop over the frames of the video
while True:
    # Read the current frame from the video
    (grabbed, frame) = camera.read()

    # Check if the frame was successfully grabbed
    if not grabbed:
        break

    # Convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge map
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Ensure that the approximated contour is roughly rectangular
        if len(approx) >= 4 and len(approx) <= 6:
            # Draw the bounding box region of the approximated contour
            (x, y, w, h) = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Compute the center of the bounding box
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

            # Draw crosshairs on the target
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 1, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
