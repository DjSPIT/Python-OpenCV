import cv2
import numpy as np
import os
from HandTracking.HandTrackingModule import HandDetector

# To create a list of the images
folder = "HeaderImages"
myList = os.listdir(folder)

overlayList = []
# Importing the images for Virtual Painter
for imagePath in myList:
    image = cv2.imread(f'{folder}/{imagePath}')
    overlayList.append(image)
# print(len(overlayList))
headerimg = overlayList[0]

# Green, Purple, Red
drawColor = (0, 255, 0)

brushStroke = 10
eraser = 60

xprev, yprev = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(min_dec_conf=0.85)

while True:
    # 1 -> Import the Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2 -> Find hand landmarks
    # Use the method we created to find the hand landmarks
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0:
        #  Index finger tip -> x1 and y1
        x1, y1 = lmList[8][1:]
        #  Middle finger tip -> x1 and y1
        x2, y2 = lmList[12][1:]

        # 3 -> Check which fingers are up -> draw with index finger, select with 2 fingers
        fingers = detector.fingers_up()
        # print(fingers)

        # 4 -> If Selection mode
        if fingers[1] and fingers[2]:
            xprev, yprev = 0, 0
            # If we are inside the header image
            if y1 < 136:
                if 200 < x1 < 400:
                    headerimg = overlayList[0]
                    drawColor = (0, 255, 0)
                elif 480 < x1 < 680:
                    headerimg = overlayList[1]
                    drawColor = (255, 0, 255)
                elif 700 < x1 < 900:
                    headerimg = overlayList[2]
                    drawColor = (0, 0, 255)
                elif 920 < x1 < 1200:
                    headerimg = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), drawColor, cv2.FILLED)

        # 5 -> If Drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)

            # This if statement suggests that the line is to be drawn from the point
            # where the finger is placed and not the origin
            if xprev == 0 and yprev == 0:
                xprev, yprev = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xprev, yprev), (x1, y1), drawColor, eraser)
                cv2.line(imgCanvas, (xprev, yprev), (x1, y1), drawColor, eraser)
            else:
                cv2.line(img, (xprev, yprev), (x1, y1), drawColor, brushStroke)
                cv2.line(imgCanvas, (xprev, yprev), (x1, y1), drawColor, brushStroke)
            xprev, yprev = x1, y1

    # Converting the image to inverse black and white to create a mask and adding the two images
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)

    # Now we have to overlay the image therefore we slice the images
    img[0:136, 0:1280] = headerimg

    # As we cannot draw on the original image, we have to add the two images
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
