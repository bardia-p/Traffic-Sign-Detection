import json
import os
import numpy as np
import cv2
import torch
from sklearn.metrics import jaccard_score
import warnings
import sys

from sign_detection.sign_detector import SignDetector

sys.path.append("..")

from main import process_image, match, template_match, sift_match
from sign_translation.sign_translator import SignTranslator

LOGGER_ENABLED = True


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


def process(data, jaccard_threshold):
    im_id = data['id']
    # print('Current image: ' + str(im_id))

    # Remap true signs to a common datatype
    real_signs = []
    for sign in data['signs']:
        real_signs += [sign]

    # Retrieve the image and run it through the process
    my_path = os.path.abspath(os.path.dirname(__file__))
    im_path = os.path.join(my_path, './input/' + str(im_id) + '.jpg')

    _, detected_signs = process_image(im_path, output_file=str(im_id))

    # Remap detected signs to a common datatype
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
            if score > jaccard_threshold:
                successful_pairs += [(i, j)]
                if found_signs[j]['sign_name'] == translator.get_sign(real_signs[i]['sign_name']):
                    successes += 1
                else:
                    failures += 1

    true_matches = len(successful_pairs)
    false_positives = len(found_signs) - len(successful_pairs)
    false_negatives = len(real_signs) - len(successful_pairs)

    return true_matches, false_positives, false_negatives, successes, failures


def print_stats(true_matches, false_positives, false_negatives, successes, failures, print_val=False):
    if not (print_val or LOGGER_ENABLED):
        return

    precision = (true_matches / (true_matches + false_positives)) \
        if (true_matches + false_positives > 0) else 0.0
    recall = (true_matches / (true_matches + false_negatives)) \
        if (true_matches + false_negatives > 0) else 0.0

    accuracy = (successes / (successes + failures)) \
        if (successes + failures > 0) else 0.0

    print("Detection Precision: " + str(precision))
    print("Detection Recall: " + str(recall))
    print("Recognition Accuracy: " + str(accuracy))


def full_eval(jaccard_threshold=0.8):
    my_path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(my_path, 'test_suite.json'))

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
        found_true_matches, false_positives, false_negatives, found_successes, found_failures = (
            process(i, jaccard_threshold))
        total_true_matches += found_true_matches
        total_false_positive += false_positives
        total_false_negative += false_negatives
        total_successes += found_successes
        total_failures += found_failures

        if counter % 10 == 0:
            print("Evaluated " + str(counter) + ' images')

        counter += 1

    return total_true_matches, total_false_positive, total_false_negative, total_successes, total_failures


def eval_jaccard(scores):

    true_positive_count = []
    false_positive_count = []
    false_negative_count = []
    for _ in scores:
        true_positive_count += [0]
        false_positive_count += [0]
        false_negative_count += [0]

    my_path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(my_path, 'test_suite.json'))

    data = json.load(f)

    counter = 1
    for image_data in data['images']:

        im_id = image_data['id']
        # print('Current image: ' + str(im_id))

        # Remap true sign locations to a common datatype
        real_signs = []
        for sign in image_data['signs']:
            real_signs += [sign]

        # Retrieve the image and run it through the process
        my_path = os.path.abspath(os.path.dirname(__file__))
        im_path = os.path.join(my_path, './input/' + str(im_id) + '.jpg')

        image = cv2.imread(im_path)

        detected_signs = SignDetector().find_signs(image.copy())

        # Remap detected signs to a common datatype
        found_signs = []
        for sign in detected_signs:
            new_dict = dict()
            new_dict['x'] = sign[1][0]
            new_dict['y'] = sign[1][1]
            new_dict['w'] = sign[1][2]
            new_dict['h'] = sign[1][3]
            found_signs += [new_dict]

        successful_pairs = []
        for _ in scores:
            successful_pairs += [[]]

        for i in range(len(real_signs)):
            for j in range(len(found_signs)):
                rec1 = [real_signs[i]['minx'], real_signs[i]['miny'], real_signs[i]['maxx'], real_signs[i]['maxy']]
                rec2 = [found_signs[j]['x'], found_signs[j]['y'], found_signs[j]['x'] + found_signs[j]['w'],
                        found_signs[j]['y'] + found_signs[j]['h']]
                for k in range(len(scores)):
                    score = get_jaccard_score(rec1, rec2)
                    if score > scores[k]:
                        successful_pairs[k] += [(i, j)]

        for k in range(len(scores)):
            true_positive_count[k] += len(successful_pairs[k])
            false_positive_count[k] += len(found_signs) - len(successful_pairs[k])
            false_negative_count[k] += len(real_signs) - len(successful_pairs[k])

        if counter % 10 == 0:
            print("Evaluated " + str(counter) + ' images')
        counter += 1


    return_dic = dict()
    for k in range(len(scores)):

        precision = (true_positive_count[k] / (true_positive_count[k] + false_positive_count[k])) \
            if (true_positive_count[k] + false_positive_count[k] > 0) else 0.0
        recall = (true_positive_count[k] / (true_positive_count[k] + false_negative_count[k])) \
            if (true_positive_count[k] + false_negative_count[k] > 0) else 0.0
        return_dic[scores[k]] = {"precision": precision, "recall": recall}
    return return_dic


def eval_accuracy():
    # Index meaning:
    # 0: all
    # 1: nn
    # 2: template match (guided by NN)
    # 3: sift           (guided by NN)
    # 4: template match
    # 5: sift
    successes_count = [0, 0, 0, 0, 0, 0]
    failures_count = [0, 0, 0, 0, 0, 0]

    my_path = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(my_path, 'test_suite.json'))

    data = json.load(f)

    counter = 1
    for image_data in data['images']:

        im_id = image_data['id']

        my_path = os.path.abspath(os.path.dirname(__file__))
        im_path = os.path.join(my_path, './input/' + str(im_id) + '.jpg')

        image = cv2.imread(im_path)

        for sign in image_data['signs']:
            x_bot = int(sign['minx'])
            x_far = int(sign['maxx'])
            y_bot = int(sign['miny'])
            y_far = int(sign['maxy'])
            sign_from_image = image[y_bot:y_far, x_bot:x_far]
            top_choice, test_results = match(sign_from_image, ['nn', 'tm', 'sift'])
            test_results += [template_match(sign_from_image)]
            test_results += [sift_match(sign_from_image)]
            print(test_results)
            print(top_choice.mode)
            actual_sign = sign['sign_name']
            if top_choice.mode == actual_sign:
                successes_count[0] += 1
            else:
                failures_count[0] += 1

            for i in range(5):
                if test_results[i] == actual_sign:
                    successes_count[i+1] += 1
                else:
                    failures_count[i+1] += 1

        if counter % 10 == 0:
            print("Evaluated " + str(counter) + ' images')
        counter += 1

    accuracy_overall = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        accuracy_overall[i] = successes_count[i] /(successes_count[i] + failures_count[i])

    return accuracy_overall

def main():
    torch.set_warn_always(False)

    warnings.filterwarnings("ignore", category=UserWarning)

    print("Beginning Detection Evaluation")
    #scores = [0.95, 0.90, 0.8, 0.6]
    #detec_result = eval_jaccard(scores)
    #for key in detec_result.keys():
    #    print("Jaccard Threshold of " + str(key) + ": Precision = " + str(detec_result[key]['precision']) +
    #          ', Recall ' + str(detec_result[key]['recall']))
    print('\n\n')

    print("Beginning Accuracy Evaluation")
    accuracy_result = eval_accuracy()
    print("Overall Accuracy: " + str(accuracy_result[0]))
    print("Neural Network Accuracy: " + str(accuracy_result[1]))
    print("Template Matching Accuracy(guided by NN): " + str(accuracy_result[2]))
    print("SIFT Accuracy(guided by NN): " + str(accuracy_result[3]))
    print("Template Matching Accuracy: " + str(accuracy_result[4]))
    print("SIFT Accuracy: " + str(accuracy_result[5]))

    print('\n\n')

    print("Beginning Overall Evaluation")
    true_match_final, false_positive_final, false_negative_final, successes_final, failures_final = full_eval()
    print_stats(true_match_final, false_positive_final, false_negative_final, successes_final, failures_final,
                print_val=True)


if __name__ == '__main__':
    main()
