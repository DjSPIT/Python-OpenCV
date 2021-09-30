# Hand Tracking using Mediapipe library
#  We have created a module file so that this can be used later in any of the project
import cv2
import mediapipe as mp
import time
import math


# We will create a class here

class HandDetector:
    def __init__(self, mode=False, max_hands=2, min_dec_conf=0.5, track_conf=0.5):

        self.mode = mode
        self.max_hands = max_hands
        self.min_dec_conf = min_dec_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        # Static mode is false to make the detection faster
        # Max number of hands is 2
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.min_dec_conf, self.track_conf)
        # To draw a line between all the hand landmarks
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Frame is processed
        self.results = self.hands.process(img_rgb)
        # Extract multiple hands
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            # handLms is a single hand
            for handLms in self.results.multi_hand_landmarks:
                # The third attribute will draw the connections for us
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # Function created to detect one particular hand position
    def find_position(self, img, hand_no=0, draw=True):
        # The list contains the idno and positions of the hand landmarks
        self.lm_list = []
        x_list = []
        y_list = []
        bbox = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for idNo, lm in enumerate(my_hand.landmark):
                #   To get the pixel value
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                # We get the centers of these hand landmarks
                self.lm_list.append([idNo, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)

        return self.lm_list, bbox

    def fingers_up(self):
        # Creating a list of fingers
        fingers = []

        # If the tip of our thumb is either on the right or on the left (open/closed)
        if self.lm_list[self.tipIds[0]][1] < self.lm_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # If the tip of the finger is above the other landmark of finger or not (open/closed)
        for idNo in range(1, 5):
            if self.lm_list[self.tipIds[idNo]][2] < self.lm_list[self.tipIds[idNo] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True, r=12, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r-4, (0, 255, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), t)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

# Now we just can use this code in any other project.
# We just need to import HandTrackingModule


def main():
    # Frame Rate
    p_time = 0
    capture = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = capture.read()
        # Method is called here
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        # Can print the value at any of the index
        if len(lm_list) != 0:
            print(lm_list[4])
            fingers = detector.fingers_up()
            print(fingers)

        curr_time = time.time()
        fps = 1 / (curr_time - p_time)
        p_time = curr_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
