import cv2
import time
import HandTrackingModule as htm

capture = cv2.VideoCapture(0)
capture.set(3, 940)
capture.set(4, 800)


def main():
    previousTime = 0

    detector = htm.HandDetector(detectionconf=0.8)

    tipids = [4, 8, 12, 16, 20]
    while True:
        success, img = capture.read()
        img = detector.findHands(img, draw=False)
        lmlist = detector.findPosition(img, draw=False)

        if len(lmlist) != 0:
            fingers = []
            # Thumb
            if lmlist[tipids[0]][1] > lmlist[tipids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 fingers
            for i in range(1, 5):
                if lmlist[tipids[i]][2] < lmlist[tipids[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalfingers = fingers.count(1)
            # print(totalfingers)
            cv2.putText(img, str(totalfingers), (45, 375), cv2.FONT_ITALIC, 10, (0, 0, 255), 20)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, f'fps: {int(fps)}', (40, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2, cv2.FILLED)

        cv2.imshow("Image", img)
        cv2.waitKey()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
