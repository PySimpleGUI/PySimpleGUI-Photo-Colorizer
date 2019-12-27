"""
    Colorization based on the Zhang Image Colorization Deep Learning Algorithm
    This header to remain with this code.

    The implementation of the colorization algorithm is from PyImageSearch
    You can learn how the algorithm works and the details of this implementation here:
    https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/

    You will need to download the pre-trained data from this location and place in the model folder:
    https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

    GUI implemented in PySimpleGUI by the PySimpleGUI group
    Of course, enjoy, learn , play, have fun!
    Copyright 2019 PySimpleGUI
"""

import numpy as np
import cv2
import PySimpleGUI as sg
import os.path

prototxt = r'model/colorization_deploy_v2.prototxt'
model = r'model/colorization_release_v2.caffemodel'
points = r'model/pts_in_hull.npy'

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
    :return: Tuple[cv2 frame, cv2 frame] both non-colorized and colorized images in cv2 format as a tuple
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
    return image, colorized

# --------------------------------- The GUI ---------------------------------

# First the window layout...2 columns

left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
            [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-')]]

images_col = [[sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
              [sg.Button('Colorize Photo', key='-PHOTO-'), sg.Button('Start Webcam', key='-WEBCAM-'), sg.Button('Save File', key='-SAVE-'), sg.Button('Exit')],
              [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],]
# ----- Full layout -----
layout = [[sg.Col(left_col), sg.VSeperator(), sg.Col(images_col)]]

# ----- Make the window -----
window = sg.Window('Photo Colorizer', layout, grab_anywhere=True)

# ----- Run the Event Loop -----
colorized = None
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == '-FOLDER-':         # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER-']
        img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
        # get list of files in folder
        try:
            flist0 = os.listdir(folder)
        except:
            continue
        fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
        window['-FILE LIST-'].update(fnames)
    elif event == '-FILE LIST-':    # A file was chosen from the listbox
        filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
        image = cv2.imread(filename)
        imgbytes_in = cv2.imencode('.png', image)[1].tobytes()
        window['-IN-'].update(data=imgbytes_in)
        window['-OUT-'].update(data='')
        window['-IN FILE-'].update('')
    elif event == '-PHOTO-':        # Colorize photo button clicked
        if values['-IN FILE-']:
            filename = values['-IN FILE-']
        elif values['-FILE LIST-']:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
        else:
            continue
        image, colorized = colorize_image(filename)
        imgbytes_in = cv2.imencode('.png', image)[1].tobytes()
        imgbytes_out = cv2.imencode('.png', colorized)[1].tobytes()
        window['-IN-'].update(data=imgbytes_in)
        window['-OUT-'].update(data=imgbytes_out)
    elif event == '-IN FILE-':      # A single filename was chosen
        filename = values['-IN FILE-']
        try:
            image = cv2.imread(filename)
            imgbytes_in = cv2.imencode('.png', image)[1].tobytes()
            window['-IN-'].update(data=imgbytes_in)
            window['-OUT-'].update(data='')
        except:
            continue
    elif event == '-WEBCAM-':       # Webcam button clicked
        window['-WEBCAM-'].update('Stop Webcam')
        cap = cv2.VideoCapture(0)
        while True:                 # Loop that reads and shows webcam until stop button
            ret, frame = cap.read()
            image, colorized = colorize_image(cv2_frame=frame)
            imgbytes_in = cv2.imencode('.png', image)[1].tobytes()
            imgbytes_out = cv2.imencode('.png', colorized)[1].tobytes()
            window['-IN-'].update(data=imgbytes_in)
            window['-OUT-'].update(data=imgbytes_out)
            event, values = window.read(timeout=0)
            if event in (None, '-WEBCAM-'): # Clicked the Stop Webcam button or closed window entirely
                window['-WEBCAM-'].update('Start Webcam')
                cap.release()
                break
    elif event == '-SAVE-' and colorized is not None:   # Clicked the Save File button
        filename = sg.popup_get_file('Save colorized image as')
        if filename:
            cv2.imwrite(filename, colorized)
        sg.popup_quick_message('Image save complete', background_color='red', text_color='white', font='Any 16')
# ----- Exit program -----
window.close()