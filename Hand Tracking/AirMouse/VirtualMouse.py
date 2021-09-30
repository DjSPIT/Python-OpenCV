import cv2
import numpy as np
import math
from HandTracking.HandTrackingModule import HandDetector
import pyautogui as apg

###################################
# Width and height of the camera
wCam, hCam = 640, 480
wScreen, hScreen = apg.size()
frameReduction = 50
smoothening = 8
###################################

cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# propId for width is 3 and for height is 4
cap.set(3, wCam)
cap.set(4, hCam)
detector = HandDetector(max_hands=1, min_dec_conf=0.85)


while True:
    # 1 -> Find the hand landmarks
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list, bbox = detector.find_position(img)

    # 2 -> Get the tip of the index and middle fingers
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        # 3 -> Check which fingers are up
        fingers = detector.fingers_up()
        cv2.rectangle(img, (frameReduction, frameReduction), (wCam - frameReduction, hCam - frameReduction),
                      (255, 0, 255), 2)

        # 4 -> Only Index Finger -> Moving Mode
        if fingers[1] == 1 and not fingers[2]:

            # 4.1 -> Convert the co-ordinates to get correct position
            x3 = np.interp(x1, (frameReduction, wCam-frameReduction), (0, wScreen))
            y3 = np.interp(y1, (frameReduction, hCam-frameReduction), (0, hScreen))

            # 4.2 -> Smoothen the values
            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            # 4.3 -> Move the mouse
            apg.moveTo(wScreen - curr_x, curr_y)
            cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y

        # 5 -> Both Index and Middle Finger -> Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, linePar = detector.find_distance(8, 12, img)
            print(length)
            # 6 -> Find the distance between the fingers and if the distance is short
            if length < 30:
                cv2.circle(img, (linePar[4], linePar[5]), 7, (0, 255, 0), cv2.FILLED)
                # 6.1 -> Clicking action
                apg.click()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
