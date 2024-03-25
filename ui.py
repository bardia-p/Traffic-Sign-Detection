import PySimpleGUI as sg
import cv2
from sign_detection.sign_detector import SignDetector
from neural_network.src.recognize_image import Recog
import time
import os
from PIL import Image, ImageTk

def process_image(image_path):
    image = cv2.imread(image_path)
    signs = SignDetector().find_signs(image.copy())
    clone = image.copy()

    for sign in signs:
        top_recogs = Recog().recog_image(sign[0])
        most_likely = top_recogs[0][1]
        cv2.rectangle(clone, (sign[1][0], sign[1][1]), (sign[1][0] + sign[1][2], sign[1][1] + sign[1][3]), (0, 255, 0), 2)
        cv2.putText(clone, str(most_likely), (sign[1][0], sign[1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    output_filename = f"processed_signs_{int(time.time())}.png"
    cv2.imwrite(output_filename, clone)
    return os.path.abspath(output_filename)

layout = [
    [sg.Text('Image File'), sg.Input(key='-FILE-'), sg.FileBrowse(), sg.Button('Process')],
    [sg.Image(key='-IMAGE-', size=(800, 450))],
    [sg.Button('Input'), sg.Button('Output')],
    [sg.Multiline(default_text='Status/Logs will appear here', size=(60, 5), key='-STATUS-')]
]

window = sg.Window('Sign Detection UI', layout)

image_path = None
processed_image_path = None

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'Process':
        image_path = values['-FILE-']
        window['-STATUS-'].update('Processing...')
        processed_image_path = process_image(image_path)
        while not os.path.exists(processed_image_path):
            time.sleep(0.1)  # Dynamic waiting for the processed image to be available
        window['-STATUS-'].update('Detection Complete')
        pil_image_processed = Image.open(processed_image_path)
        window['-IMAGE-'].update(data=ImageTk.PhotoImage(pil_image_processed))
    elif event == 'Input':
        if image_path:
            pil_image = Image.open(os.path.abspath(image_path))
            window['-IMAGE-'].update(data=ImageTk.PhotoImage(pil_image))
    elif event == 'Output':
        if processed_image_path:
            pil_image_processed = Image.open(processed_image_path)
            window['-IMAGE-'].update(data=ImageTk.PhotoImage(pil_image_processed))

window.close()