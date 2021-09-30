# Thank you AndreMiras for this amazing library to control volume of our computer using python

import cv2
import numpy as np
import math
from HandTracking.HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

###################################
# Width and height of the camera
wCam, hCam = 720, 720
###################################

cap = cv2.VideoCapture(0)
# propId for width is 3 and for height is 4
cap.set(3, wCam)
cap.set(4, hCam)

# Calling the class as an object to use in this program
detector = HandDetector(min_dec_conf=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volBar = 0
volPer = 0
# Now we need to convert our volume ranges
# Volume is at 0%
# volume.SetMasterVolumeLevel(-65.0, None)
# Volume is at 26%
# volume.SetMasterVolumeLevel(-20.0, None)
# Volume is at 72%
# volume.SetMasterVolumeLevel(-5.0, None)
# # Volume is at 100%
# volume.SetMasterVolumeLevel(-0.0, None)

while True:
    success, img = cap.read()

    # Use the method we created to find the hand landmarks
    img = detector.find_hands(img)

    # Get the position of the landmarks
    lmList = detector.find_position(img, draw=False)
    if len(lmList) != 0:
        # print("Landmark 4 (Thumb): ", lmList[4], "\nLandmark 8 (Index finger): ", lmList[8], "\n")

        # X and Y co-ordinates of thumb and index finger tip
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw a circle around those two point and a line connecting them
        cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # We need to find the length of the line created between the two points
        length = math.hypot(x2 - x1, y2 - y1)
        # Max and Min length -> 300 and 15
        # Volume range -> -65 to 0
        # Therefore we have a function in numpy
        volComputer = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print("Length: ", int(length), "Volume:", volComputer)
        volume.SetMasterVolumeLevel(volComputer, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'Vol:{int(volPer)}%', (40, 440), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
