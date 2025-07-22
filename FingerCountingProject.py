import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import HandTrackingModule as htm 

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0) 
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "finger_images" 


myList = os.listdir(folderPath)
# Sort by converting filename (without extension) to int
myList.sort(key=lambda f: int(os.path.splitext(f)[0]))
print(myList) 

overlayList = []
for imPath in myList:
    image_path = os.path.join(folderPath, imPath)
    image = cv2.imread(image_path)
    target_overlay_size = (200, 200) # Adjust as needed for your UI
    resized_overlay = cv2.resize(image, target_overlay_size, interpolation=cv2.INTER_AREA)
    overlayList.append(resized_overlay)

print(f"Number of overlay images loaded: {len(overlayList)}")

pTime = 0 
detector = htm.handDetector(maxHands=2, detectionCon=0.75)

# Landmark IDs for the tips of fingers:
# Thumb: 4
# Index: 8
# Middle: 12
# Ring: 16
# Pinky: 20
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read() 
    img = cv2.flip(img, 1) 

    img = detector.findHands(img, draw=True) 
    all_lmLists, all_handedness = detector.findPosition(img, draw=True) 
    totalFingersUp = 0 

    for i, lmList in enumerate(all_lmLists):
        current_hand_label = all_handedness[i] if i < len(all_handedness) else "Unknown"
        fingers_current_hand = []
        # --- LOGIC FOR THUMB ---
        # Check if the hand is right or left to determine thumb position        
        if current_hand_label == "Right":
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]: # Thumb tip X > IP joint X
                fingers_current_hand.append(1)
            else:
                fingers_current_hand.append(0)
        elif current_hand_label == "Left":
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]: # Thumb tip X < IP joint X
                fingers_current_hand.append(1)
            else:
                fingers_current_hand.append(0)
        else: # Unknown handedness, default to 0 for thumb
            fingers_current_hand.append(0)

        for id in range(1, 5): # Loop for Index (1), Middle (2), Ring (3), Pinky (4)
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers_current_hand.append(1) # Finger is open
            else:
                fingers_current_hand.append(0) # Finger is closed

        # Sum up fingers for the current hand
        totalFingersUp += fingers_current_hand.count(1)

    if totalFingersUp >= 0 and totalFingersUp < len(overlayList):
        h_overlay, w_overlay, c_overlay = overlayList[totalFingersUp].shape
        img[0:h_overlay, 0:w_overlay] = overlayList[totalFingersUp]
    else:
        print(f"Warning: totalFingersUp ({totalFingersUp}) out of range for overlayList (size {len(overlayList)}).")
        if len(overlayList) > 0:
            h_overlay, w_overlay, c_overlay = overlayList[-1].shape 
            img[0:h_overlay, 0:w_overlay] = overlayList[-1]

    cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingersUp), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 0), 3)

    cv2.imshow("Image", img) 

    key = cv2.waitKey(1) 
    if key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()