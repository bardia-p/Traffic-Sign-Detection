import cv2

from sign_detection.sign_detector import SignDetector

from neural_network.src.recognize_image import Recog

image = cv2.imread("test.jpg")

signs = SignDetector().find_signs(image.copy())

clone = image.copy()

for sign in signs:
    top_recogs = Recog().recog_image(sign[0])

    most_likely = top_recogs[0][1]

    cv2.rectangle(clone, (sign[1][0], sign[1][1]), (sign[1][0] + sign[1][2], sign[1][1] + sign[1][3]), (0, 255, 0), 2)
    cv2.putText(clone, str(most_likely), (sign[1][0], sign[1][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imwrite("owo.png", clone)
