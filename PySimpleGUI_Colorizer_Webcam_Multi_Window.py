"""
    July 2020 - experimental multi-window version of the webcam portion of window colorizer program

    Colorization based on the Zhang Image Colorization Deep Learning Algorithm
    This header to remain with this code.

    The implementation of the colorization algorithm is from PyImageSearch
    You can learn how the algorithm works and the details of this implementation here:
    https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/

    You will need to download the pre-trained data from this location and place in the model folder:
    https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

    GUI implemented in PySimpleGUI by the PySimpleGUI group
    Of course, enjoy, learn , play, have fun!
    Copyright 2020 PySimpleGUI
"""

import numpy as np
import cv2
import PySimpleGUI as sg
import os.path

prototxt = r'model/colorization_deploy_v2.prototxt'
model = r'model/colorization_release_v2.caffemodel'
points = r'model/pts_in_hull.npy'
points = os.path.join(os.path.dirname(__file__), points)
prototxt = os.path.join(os.path.dirname(__file__), prototxt)
model = os.path.join(os.path.dirname(__file__), model)
if not os.path.isfile(model):
    sg.popup_scrolled('Missing model file', 'You are missing the file "colorization_release_v2.caffemodel"',
                      'Download it and place into your "model" folder', 'You can download this file from this location:\n', r'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1')
    exit()
net = cv2.dnn.readNetFromCaffe(prototxt, model)     # load model from disk
pts = np.load(points)

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_filename=None, cv2_frame=None):
    """
    Where all the magic happens.  Colorizes the image provided. Can colorize either
    a filename OR a cv2 frame (read from a web cam most likely)
    :param image_filename: (str) full filename to colorize
    :param cv2_frame: (cv2 frame)
    :return: cv2 frame colorized image in cv2 format
    """
    # load the input image from disk, scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    image = cv2.imread(image_filename) if image_filename else cv2_frame
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    'print("[INFO] colorizing image...")'
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # grab the 'L' channel from the *original* input image (not the resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # the current colorized image is represented as a floating point data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    return  colorized


def convert_to_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert webcam frame to grayscale
    gray_3_channels = np.zeros_like(frame)  # Convert grayscale frame (single channel) to 3 channels
    gray_3_channels[:, :, 0] = gray
    gray_3_channels[:, :, 1] = gray
    gray_3_channels[:, :, 2] = gray
    return gray_3_channels


def make_video_window(title, location):
    return sg.Window(title, [[sg.Image(key='-IMAGE-')]], finalize=True, margins=(0,0), element_padding=(0,0), location=location)

def convert_cvt_to_data(cv2_frame):
    return cv2.imencode('.png', cv2_frame)[1].tobytes()


def main():
    # --------------------------------- The GUI ---------------------------------

    layout = [  [sg.Text('Colorized Webcam Demo', font='Any 18')],
                [sg.Button('Start Webcam', key='-WEBCAM-'), sg.Button('Exit')]]

    # ----- Make the starting window -----
    window_start = sg.Window('Webcam Colorizer', layout, grab_anywhere=True, finalize=True)

    # ----- Run the Event Loop -----
    cap, playback_active = None, False
    while True:
        window, event, values = sg.read_all_windows(timeout=10)
        if event == 'Exit' or (window == window_start and event is None):
            break
        elif event == '-WEBCAM-':       # Webcam button clicked
            if not playback_active:
                sg.popup_quick_message('Starting up your Webcam... this takes a moment....', auto_close_duration=1,  background_color='red', text_color='white', font='Any 16')
                window_start['-WEBCAM-'].update('Stop Webcam', button_color=('white','red'))
                cap = cv2.VideoCapture(0) if not cap else cap
                window_raw_camera = make_video_window('Your Webcam Raw Video', (300,200))
                window_gray_camera = make_video_window('Video as Grayscale', (1000,200))
                window_colorized_camera = make_video_window('Your Colorized Video', (1700,200))
                playback_active = True
            else:
                playback_active = False
                window['-WEBCAM-'].update('Start Webcam', button_color=sg.theme_button_color())
                window_raw_camera.close()
                window_gray_camera.close()
                window_colorized_camera.close()
        elif event == sg.TIMEOUT_EVENT and playback_active:
            ret, frame = cap.read()  # Read a webcam frame

            # display raw image
            if window_raw_camera:
                window_raw_camera['-IMAGE-'].update(data=convert_cvt_to_data(frame))
            # display gray image
            gray_3_channels = convert_to_grayscale(frame)
            if window_gray_camera:
                window_gray_camera['-IMAGE-'].update(data=convert_cvt_to_data(gray_3_channels))
            # display colorized image
            if window_colorized_camera:
                window_colorized_camera['-IMAGE-'].update(data=convert_cvt_to_data(colorize_image(cv2_frame=gray_3_channels)))

        # if a window closed
        if event is None:
            if window == window_raw_camera:
                window_raw_camera.close()
                window_raw_camera = None
            elif window == window_gray_camera:
                window_gray_camera.close()
                window_gray_camera = None
            elif window == window_colorized_camera:
                window_colorized_camera.close()
                window_colorized_camera = None

        # If playback is active, but all camera windows closed, indicate not longer playing and change button color
        if playback_active and window_colorized_camera is None and window_gray_camera is None and window_raw_camera is None:
            playback_active = False
            window_start['-WEBCAM-'].update('Start Webcam', button_color=sg.theme_button_color())

    # ----- Exit program -----
    window.close()

if __name__ == '__main__':
    main()

