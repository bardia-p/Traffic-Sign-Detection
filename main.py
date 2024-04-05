import cv2
import random
from datetime import datetime
from scipy.stats import mode
import os
import json

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

    translator = SignTranslator()

    results = dict()

    for sign in signs:
        test_results = []
        gray_sign = cv2.cvtColor(sign[0], cv2.COLOR_BGR2GRAY)

        top_recogs = Recog().recog_image(sign[0])
        if len(top_recogs) > 0:
            test_results.append(top_recogs[0][1])
    
        tms = TemplateMatcher().template_match(gray_sign, top_recogs)

        if len(tms) > 0:
            test_results.append(tms[0][1])

        sifts = TemplateMatcher().sift_match(gray_sign, top_recogs)

        if len(sifts) > 0:
            test_results.append(sifts[0][1])

        #test_resultsprint()
        top_choice = mode(test_results)

        if top_choice.count != 0:
            sign_details = dict()
            sign_details["x"] = sign[1][0]
            sign_details["y"] = sign[1][1]
            sign_details["w"] = sign[1][2]
            sign_details["h"] = sign[1][3]

            results[translator.get_sign(top_choice.mode)] = sign_details

    for r in results.keys():
        sign = results[r]
        cv2.rectangle(image,(sign["x"], sign["y"]), (sign["x"] + sign["w"], sign["y"] + sign["h"]), (0, 255, 0), 2)
        cv2.putText(image, r, (sign["x"], sign["y"]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if output_file == "":
        output_file = "processed_signs_" + str(random.randint(1,1000))

    output_dir =  OUTPUT_DIR + output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_image = output_dir + "/" + output_file + ".png"
    cv2.imwrite(output_image, image)

    output_file = output_dir + "/" + output_file + ".json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(results)
    return output_image, results

if __name__ == '__main__':
    process_image(INPUT_DIR + "test.jpg", "result")