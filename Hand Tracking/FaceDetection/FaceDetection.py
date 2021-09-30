# Face Detection basics
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
# Draw something for detection
mpDraw = mp.solutions.drawing_utils
# Initialize the face detection module
faceDetection = mpFaceDetection.FaceDetection()

# Detections
# 0 label_id: 0
# score: 0.9525054693222046
# location_data {
#   format: RELATIVE_BOUNDING_BOX
#   relative_bounding_box {
#     xmin: 0.3900827467441559
#     ymin: 0.3853267431259155
#     width: 0.3276645839214325
#     height: 0.4368742108345032
#   }
#   relative_keypoints {
#     x: 0.4581410586833954
#     y: 0.515761137008667
#   }
#   relative_keypoints {
#     x: 0.5958595275878906
#     y: 0.5115149617195129
#   }
#   relative_keypoints {
#     x: 0.5111615657806396
#     y: 0.6282121539115906
#   }
#   relative_keypoints {
#     x: 0.5210227370262146
#     y: 0.7102954387664795
#   }
#   relative_keypoints {
#     x: 0.4157966375350952
#     y: 0.5442922115325928
#   }
#   relative_keypoints {
#     x: 0.7028709053993225
#     y: 0.5368989109992981
#   }
# }


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for idNo, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)

            # This is a very long call
            # We need to get the information about the bounding box
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            # We are manually drawing the rectangle on the face for detection
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
