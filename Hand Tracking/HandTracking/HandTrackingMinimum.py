# Hand Tracking using Mediapipe library
#  We have created a module file so that this can be used later in any of the project
import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
# Static mode is false to make the detection faster
# Max number of hands is 2
hands = mpHands.Hands()
# To draw a line between all the hand landmarks
mpDraw = mp.solutions.drawing_utils

# Frame Rate
pTime = 0
currTime = 0

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Frame is processed
    results = hands.process(imgRGB)
    # Extract multiple hands
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        # handLms is a single hand
        for handLms in results.multi_hand_landmarks:
            for idNo, lm in enumerate(handLms.landmark):
                #   To get the pixel value
                h, w, c = img.shape
                cX, cY = int(lm.x*w), int(lm.y*h)
                # We get the centers of these hand landmarks
                print(idNo, ":", cX, cY)
                if idNo == 4:
                    cv2.circle(img, (cX, cY), 20, (255, 0, 255), cv2.FILLED)
                if idNo == 8:
                    cv2.circle(img, (cX, cY), 20, (255, 0, 255), cv2.FILLED)

            # The third attribute will draw the connections for us
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    currTime = time.time()
    fps = 1/(currTime-pTime)
    pTime = currTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
