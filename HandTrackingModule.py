import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxhands=2, detectionconf=0.5, trackingconf=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectionconf = detectionconf
        self.trackingconf = trackingconf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands, self.detectionconf, self.trackingconf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for eachhand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, eachhand, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, landmarks in enumerate(myhand.landmark):
                height, width, channel = img.shape
                cx, cy = int(landmarks.x * width), int(landmarks.y * height)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
        return lmlist


def main():
    previousTime = 0
    capture = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = capture.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        currentTime = time.time()
        fps = 30 / (currentTime - previousTime)  # 30 frames per second
        previousTime = currentTime

        cv2.putText(img, "fps: "+str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 3)  # adding seconds to the frame

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
