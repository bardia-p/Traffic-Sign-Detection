import cv2

from sign_detection.sign_detector import SignDetector

from neural_network.src.recognize_image import Recog

from template_match.template_matcher import TemplateMatcher

image = cv2.imread("test.jpg")

signs = SignDetector().find_signs(image.copy())

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

results = dict()

for sign in signs:
    top_recogs = Recog().recog_image(sign[0])

    matches = TemplateMatcher().locate_sign(img_gray, top_recogs)

    if len(matches) > 0:
        tm_res = matches[0][1]
    else:
        tm_res = -1

    print(top_recogs)
    print(matches)

    if tm_res != -1:
        if tm_res not in results:
            results[tm_res] = sign

for r in results.keys():
    sign = results[r]
    cv2.rectangle(image,(sign[1][0], sign[1][1]), (sign[1][0] + sign[1][2], sign[1][1] + sign[1][3]), (0, 255, 0), 2)
    cv2.putText(image, str(r), (sign[1][0], sign[1][1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imwrite("result.png", image)