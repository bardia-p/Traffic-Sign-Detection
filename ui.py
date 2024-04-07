import PySimpleGUI as sg
import cv2
from sign_detection.sign_detector import SignDetector
from neural_network.src.recognize_image import Recog
import time
import os
from PIL import Image, ImageTk
from main import process_image

layout = [
    [sg.Text('Image File'), sg.Input(key='-FILE-'), sg.FileBrowse(), sg.Button('Process')],
    [sg.Image(key='-IMAGE-', size=(800, 450))],
    [sg.Button('Input'), sg.Button('Output')],
    [sg.Multiline(default_text='Status/Logs will appear here', size=(60, 5), key='-STATUS-')]
]

window = sg.Window('Sign Detection UI', layout, resizable=True, size=(800, 600))

image_path = None
processed_image_path = None

def resize_image(image_path, max_size=(800, 450)):
    image = Image.open(image_path)
    image.thumbnail(max_size)
    return image

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'Process':
        image_path = values['-FILE-']
        window['-STATUS-'].update('Processing...')
        processed_image_path, results = process_image(image_path)
        while not os.path.exists(processed_image_path):
            time.sleep(0.1)  # Dynamic waiting for the processed image to be available
        window['-STATUS-'].update('Detection Complete')
        pil_image_processed = resize_image(processed_image_path)
        window['-IMAGE-'].update(data=ImageTk.PhotoImage(pil_image_processed))
    elif event == 'Input':
        if image_path:
            pil_image = resize_image(os.path.abspath(image_path))
            window['-IMAGE-'].update(data=ImageTk.PhotoImage(pil_image))
    elif event == 'Output':
        if processed_image_path:
            pil_image_processed = resize_image(processed_image_path)
            window['-IMAGE-'].update(data=ImageTk.PhotoImage(pil_image_processed))

window.close()