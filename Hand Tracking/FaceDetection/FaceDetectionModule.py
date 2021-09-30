# Face Detection as a module
import cv2
import mediapipe as mp


def fancy_draw(img, bbox, l=30, t=6, rt=1):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    cv2.rectangle(img, bbox, (0, 255, 0), rt)
    # Top Left Corner
    cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
    cv2.line(img, (x, y), (x, y + l), (0, 255, 0), t)

    # Top Right Corner
    cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
    cv2.line(img, (x1, y), (x1, y + l), (0, 255, 0), t)

    # Bottom Left Corner
    cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
    cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)

    # Bottom Right Corner
    cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
    cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)
    return img


class FaceDetector:
    def __init__(self, min_detec_conf=0.5):

        self.min_detec_conf = min_detec_conf

        self.mpFaceDetection = mp.solutions.face_detection

        # Draw something for detection
        self.mpDraw = mp.solutions.drawing_utils

        # Initialize the face detection module
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detec_conf)

    # Defined a method find_faces() to detect one face at a time
    # -> Converts the image to RGB
    # -> processes the image using faceDetection module
    # -> gets the box to be drawn around the detected face
    # -> draws the box
    def find_faces(self, img, draw=True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img_rgb)

        bbox_tool = []

        if self.results.detections:
            for idNo, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection)
                # This is a very long call
                # print(detection.location_data.relative_bounding_box)
                # We need to get the information about the bounding box
                bboxc = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxc.xmin * w), int(bboxc.ymin * h), int(bboxc.width * w), int(bboxc.height * h)
                bbox_tool.append([bbox, detection.score])

                if draw:
                    # We are manually drawing the rectangle on the face for detection
                    img = fancy_draw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img, bbox_tool


def main():
    # Capturing the video
    cap = cv2.VideoCapture(0)
    # Calling the class
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        # Calling the find_faces() method of the class for the image
        img, bbox_tool = detector.find_faces(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
