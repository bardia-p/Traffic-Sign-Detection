import json
import os
import numpy as np
import cv2
from sklearn.metrics import jaccard_score
from main import process_image
from sign_translator.sign_translator import SignTranslator

JACCARD_THRESHOLD = 0.8


# Used to evaluate the threshold of tollerance for rectangle comparisons
def get_jaccard_score(rec1, rec2):
    min_x = min(rec1[0], rec2[0])
    min_y = min(rec1[1], rec2[1])
    rec1[0] = rec1[0] - min_x
    rec1[2] = rec1[2] - min_x
    rec2[0] = rec2[0] - min_x
    rec2[2] = rec2[2] - min_x

    rec1[1] = rec1[1] - min_y
    rec1[3] = rec1[3] - min_y
    rec2[1] = rec2[1] - min_y
    rec2[3] = rec2[3] - min_y

    if rec1[0] > rec2[2] or rec2[0] > rec1[2]:
        return 0.0
    if rec1[1] > rec2[3] or rec2[1] > rec1[3]:
        return 0.0

    shape = (max(rec1[3], rec2[3]) + 1, max(rec1[2], rec2[2]) + 1)
    img = np.zeros(shape, np.uint8)  # use your image shape here or directly below
    img1 = cv2.rectangle(np.zeros(img.shape), (rec1[0], rec1[1]), (rec1[2], rec1[3]), (1, 1, 1), -1)
    img2 = cv2.rectangle(np.zeros(img.shape), (rec2[0], rec2[1]), (rec2[2], rec2[3]), (1, 1, 1), -1)
    return jaccard_score(img1.ravel(), img2.ravel())


def process(data):
    im_id = data['id']
    print('Current image: ' + str(im_id))
    real_signs = []
    for sign in data['signs']:
        real_signs += [sign]

        # new_dict['minx'] = sign['minx']
        # new_dict['miny'] = sign['miny']
        # new_dict['maxx'] = sign['maxx']
        # new_dict['maxy'] = sign['maxy']
        # new_dict['sign_name'] = sign['sign_name']

    # Retrieve the image and run it through the process
    my_path = os.path.abspath(os.path.dirname(__file__))
    im_path = os.path.join(my_path, './input/' + str(im_id) + '.jpg')
    image = cv2.imread(im_path)

    _, detected_signs = process_image(im_path, output_file=str(im_id))

    found_signs = []
    for sign in detected_signs.keys():
        new_dict = dict()
        new_dict['sign_name'] = sign
        new_dict['x'] = detected_signs[sign]['x']
        new_dict['y'] = detected_signs[sign]['y']
        new_dict['w'] = detected_signs[sign]['w']
        new_dict['h'] = detected_signs[sign]['h']
        found_signs += [new_dict]

    successful_pairs = []
    successes = 0
    failures = 0
    translator = SignTranslator()

    for i in range(len(real_signs)):
        for j in range(len(found_signs)):
            rec1 = [real_signs[i]['minx'], real_signs[i]['miny'], real_signs[i]['maxx'], real_signs[i]['maxy']]
            rec2 = [found_signs[j]['x'], found_signs[j]['y'], found_signs[j]['x'] + found_signs[j]['w'],
                    found_signs[j]['y'] + found_signs[j]['h']]
            score = get_jaccard_score(rec1, rec2)
            if score > JACCARD_THRESHOLD:
                successful_pairs += [(i, j)]
                if found_signs[j]['sign_name'] == translator.get_sign(real_signs[i]['sign_name']):
                    successes += 1
                else:
                    failures += 1

    true_matches = len(successful_pairs)
    false_positives = len(found_signs) - len(successful_pairs)
    false_negatives = len(real_signs) - len(successful_pairs)

    return true_matches, false_positives, false_negatives, successes, failures


def run_eval():
    my_path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(my_path, 'test_suite.json'))

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterate through each image
    total_true_matches = 0
    total_false_positive = 0
    total_false_negative = 0
    total_successes = 0
    total_failures = 0
    print(len(data['images']))
    counter = 1
    for i in data['images']:
        found_true_matches, false_positives, false_negatives, found_successes, found_failures = process(i)
        total_true_matches += found_true_matches
        total_false_positive += false_positives
        total_false_negative += false_negatives
        total_successes += found_successes
        total_failures += found_failures

        if counter % 10 == 0:
            precision_temp = (total_true_matches / (total_true_matches + total_false_positive)) \
                if (total_true_matches + false_positive > 0) else 0.0
            recall_temp = (total_true_matches / (total_true_matches + total_false_negative)) \
                if (total_true_matches + total_false_negative > 0) else 0.0

            accuracy_temp = (total_successes / (total_successes + total_failures)) \
                if (total_successes + total_failures > 0) else 0.0

            print("True Positives:" + str(total_true_matches))
            print("False Positives:" + str(total_false_positive))
            print("False Negatives:" + str(total_false_negative))

            print("Correct Recognition:" + str(total_successes))
            print("Incorrect Recognition:" + str(total_failures))

            print("Detection Precision: " + str(precision_temp))
            print("Detection Recall: " + str(recall_temp))
            print("Recognition Accuracy: " + str(accuracy_temp))

        counter += 1

    return total_true_matches, total_false_positive, total_false_negative, total_successes, total_failures


if __name__ == '__main__':
    true_match, false_positive, false_negative, successes, failures = run_eval()

    precision = (true_match / (true_match + false_positive)) \
        if (true_match + false_positive > 0) else 0.0
    recall = (true_match / (true_match + false_negative)) \
        if (true_match + false_negative > 0) else 0.0

    accuracy = (successes / (successes + failures)) \
        if (successes + failures > 0) else 0.0

    print("True Positives:" + str(true_match))
    print("False Positives:" + str(false_positive))
    print("False Negatives:" + str(false_negative))

    print("Correct Recognition:" + str(successes))
    print("Incorrect Recognition:" + str(failures))

    print("Detection Precision: " + str(precision))
    print("Detection Recall: " + str(recall))
    print("Recognition Accuracy: " + str(accuracy))
