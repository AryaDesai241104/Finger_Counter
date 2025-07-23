import os
import cv2
import time
import HandTrackingModule as htm

# Suppress TensorFlow/MediaPipe logging messages for cleaner output
os.environ['GLOG_minloglevel'] = '2'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

wCam, hCam = 640, 480

# Initialize video capture
# The index 0 refers to the default camera. If you have multiple cameras,
# you might need to try 1, 2, etc.
cap = cv2.VideoCapture(0)
cap.set(3, wCam) # Set width
cap.set(4, hCam) # Set height

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream. Please check if the camera is connected and not in use by another application.")
    exit() # Exit the script if camera cannot be opened

folderPath = "finger_images"

# Get list of image files and sort them numerically
myList = os.listdir(folderPath)
myList.sort(key=lambda f: int(os.path.splitext(f)[0]))
print(f"Found image files: {myList}")

overlayList = []
target_overlay_size = (200, 200) # Define a consistent size for overlay images

# Load and resize overlay images
for imPath in myList:
    image_path = os.path.join(folderPath, imPath)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}. Skipping.")
        continue
    resized_overlay = cv2.resize(image, target_overlay_size, interpolation=cv2.INTER_AREA)
    overlayList.append(resized_overlay)

print(f"Number of overlay images loaded: {len(overlayList)}")

pTime = 0 # Previous time for FPS calculation

# Initialize hand detector
detector = htm.handDetector(maxHands=2, detectionCon=0.75)

# Landmark IDs for the tips of fingers (MediaPipe Hand Landmarks)
# Thumb: 4
# Index: 8
# Middle: 12
# Ring: 16
# Pinky: 20
tipIds = [4, 8, 12, 16, 20]

while True:
    # Read a frame from the camera
    success, img = cap.read()

    # IMPORTANT: Check if the frame was successfully read
    if not success:
        print("Failed to read frame from camera. Exiting...")
        break # Exit the loop if frame reading fails

    # Flip the image horizontally for a natural mirror effect
    img = cv2.flip(img, 1)

    # Find hands in the image and draw landmarks
    img = detector.findHands(img, draw=True)
    # Get landmark positions and handedness (left/right)
    all_lmLists, all_handedness = detector.findPosition(img, draw=True)
    totalFingersUp = 0

    # Process each detected hand
    for i, lmList in enumerate(all_lmLists):
        current_hand_label = all_handedness[i] if i < len(all_handedness) else "Unknown"
        fingers_current_hand = []

        # --- LOGIC FOR THUMB ---
        # The thumb's open/closed state depends on handedness (left/right)
        if current_hand_label == "Right":
            # For right hand, thumb is open if tip (4) is to the right of the IP joint (3)
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers_current_hand.append(1) # Thumb is open
            else:
                fingers_current_hand.append(0) # Thumb is closed
        elif current_hand_label == "Left":
            # For left hand, thumb is open if tip (4) is to the left of the IP joint (3)
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers_current_hand.append(1) # Thumb is open
            else:
                fingers_current_hand.append(0) # Thumb is closed
        else: # Unknown handedness, default to closed for thumb
            fingers_current_hand.append(0)

        # --- LOGIC FOR OTHER FINGERS (Index, Middle, Ring, Pinky) ---
        # These fingers are open if their tip (e.g., 8 for index) is higher (smaller Y-coordinate)
        # than the base of the finger (e.g., 6 for index)
        for id in range(1, 5): # Loop for Index (1), Middle (2), Ring (3), Pinky (4)
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers_current_hand.append(1) # Finger is open
            else:
                fingers_current_hand.append(0) # Finger is closed

        # Sum up fingers that are open for the current hand
        totalFingersUp += fingers_current_hand.count(1)

    # Display the corresponding overlay image based on totalFingersUp
    if totalFingersUp >= 0 and totalFingersUp < len(overlayList):
        # Ensure the overlay fits within the image bounds
        h_overlay, w_overlay, c_overlay = overlayList[totalFingersUp].shape
        # Place the overlay at the top-left corner
        img[0:h_overlay, 0:w_overlay] = overlayList[totalFingersUp]
    else:
        # Handle cases where totalFingersUp is out of expected range
        print(f"Warning: totalFingersUp ({totalFingersUp}) out of range for overlayList (size {len(overlayList)}).")
        if len(overlayList) > 0:
            # If out of range, display the last image in the list (e.g., for 10 fingers or more)
            h_overlay, w_overlay, c_overlay = overlayList[-1].shape
            img[0:h_overlay, 0:w_overlay] = overlayList[-1]
        else:
            print("Error: No overlay images loaded.")

    # Draw a rectangle and display the total number of fingers up
    cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingersUp), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 0, 0), 25)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 0), 3)

    # Display the final image
    cv2.imshow("Image", img)

    # Wait for a key press (1ms delay) and break loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
