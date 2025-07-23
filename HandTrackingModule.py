import cv2
import mediapipe as mp
import time
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        if img is None:
            self.results = None
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = img

        imgRGB = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if imgRGB.dtype != np.uint8:
            imgRGB = imgRGB.astype(np.uint8)

        imgRGB_processed = np.ascontiguousarray(imgRGB)

        self.results = self.hands.process(imgRGB_processed)

        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        all_lmLists = []
        all_handedness = []

        if self.results and self.results.multi_hand_landmarks:
            for i, handLms in enumerate(self.results.multi_hand_landmarks):
                lmList = []
                handedness_label = self.results.multi_handedness[i].classification[0].label if self.results.multi_handedness else "Unknown"
                all_handedness.append(handedness_label)

                for id, lm in enumerate(handLms.landmark):
                    if img is not None and len(img.shape) >= 2:
                        h, w = img.shape[:2]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy, lm.z])

                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                all_lmLists.append(lmList)

        return all_lmLists, all_handedness

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream for HandTrackingModule example.")
        return

    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame in HandTrackingModule example. Exiting.")
            break

        img = detector.findHands(img)
        lmList, handedness = detector.findPosition(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
