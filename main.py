import cv2

from sign_detection.sign_detector import SignDetector

from neural_network.src.recognize_image import Recog

from template_match.template_matcher import TemplateMatcher

image = cv2.imread("test.jpg")

signs = SignDetector().find_signs(image.copy())

clone = image.copy()

for sign in signs:
    matches = TemplateMatcher().locate_sign(sign[0])
    
    for m in matches:
        top_recogs = Recog().recog_image(m[0])

        print(top_recogs)
        most_likely = top_recogs[0][1]

        #    detections = tm.locate_sign(image, most_likely)

        #   print(most_likely, len(detections))
        #  if len(detections) > 0:
    
        cv2.rectangle(clone,(sign[1][0], sign[1][1]), (sign[1][0] + sign[1][2], sign[1][1] + sign[1][3]), (0, 255, 0), 2)
        cv2.putText(clone, str(most_likely), (sign[1][0], sign[1][1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imwrite("result.png", clone)