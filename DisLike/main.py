import cv2
import time
import os

import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.75)
totalFingers = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if lmList:
        fingersUp = detector.LikeDislike()
        totalFingers = fingersUp
    else:
        totalFingers = "0"

    cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
