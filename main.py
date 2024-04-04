import cv2
import random
from datetime import datetime

from sign_detection.sign_detector import SignDetector

from neural_network.src.recognize_image import Recog

from template_match.template_matcher import TemplateMatcher

from sign_translator.sign_translator import SignTranslator

INPUT_DIR = "inputs/"
OUTPUT_DIR = "results/"

def process_image(input_file, output_file=""):
    '''
    Processes the input image to find the sign.

    @param input_file: the name of the input image.
    @param output_file: the name of the output image.

    @return the file name for the output image.
    '''
    random.seed(datetime.now().timestamp())

    image = cv2.imread(input_file)

    signs = SignDetector().find_signs(image.copy())

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    translator = SignTranslator()

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
        cv2.putText(image, translator.get_sign(r), (sign[1][0], sign[1][1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if output_file == "":
        output_file = "processed_signs_" + str(random.randint(1,1000))

    output_file = "results/" + output_file + ".png"
    cv2.imwrite(output_file, image)

    return output_file

if __name__ == '__main__':
    process_image(INPUT_DIR + "test.jpg", "result")