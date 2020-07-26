![pysimplegui_logo](https://user-images.githubusercontent.com/13696193/43165867-fe02e3b2-8f62-11e8-9fd0-cc7c86b11772.png)

# Photo Colorization Using Deep Learning

## Complete Python Application With GUI Using PySimpleGUI

### ___Important___

In order to run the demo, you will first need to download the pre-trained data from this location. At 125 MB it's too large to put into the GitHub.  Place the file in the model folder.  

https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

-----------------

![SNAG-0613](https://user-images.githubusercontent.com/46163555/71523947-43c03a00-2899-11ea-8943-e8db1347c7f5.jpg)
![SNAG-0604](https://user-images.githubusercontent.com/46163555/71523948-4458d080-2899-11ea-8a8a-d54fbf39c9b8.jpg)

-----------------

## The Zhang Algorithm

The colorization algorithm was developed by Zhang, et al, and is detailed here:

http://richzhang.github.io/colorization/

## PyImageSearch

The code implementing the algorithm is explained in a well-written tutorial by the fine folks at PyImageSearch:

https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/


## Using the GUI

To use the GUI you'll need to install PySimpleGUI (http://www.PySimpleGUI.org for instructions)

One of these will install it for you.
```
pip install PySimpleGUI
pip3 install PySimpleGUI
```

Then you run the demo program using either `python` or `python3` dependind on your system:

```
python PySimpleGUI_Colorizer.py
python3 PySimpleGUI_Colorizer.py
```

### You have 2 options for choosing the image to colorize.

#### Folder View

If you choose a folder in the left column, then a list of files will be shown.  Clicking on a file will "Preview" the image on the right side.  Either copy and paste a path into the input box in the upper left corner, or use the `Browse` button to browse for a folder

![SNAG-0627](https://user-images.githubusercontent.com/46163555/71523944-43c03a00-2899-11ea-8dea-a3be3bfc13ca.jpg)

#### Individual File

You can also choose an individual file using the input box in the upper right.  Either paste a filename into the box or use the `browse` button to choose one.

### Webcam

Press the `Start Webcam` button to see yourself colorized in realtime. It's not super fast, but it does function.

Press the `Stop Webcam` button to stop.

## Saving The Color Image

To save your image simply press the `Save File` button and enter your filename.


-------------------------------

Here is more eye-candy courtesy of Deep Learning





![SNAG-0628](https://user-images.githubusercontent.com/46163555/71523943-4327a380-2899-11ea-95b7-a2892f611109.jpg)

![SNAG-0626](https://user-images.githubusercontent.com/46163555/71523945-43c03a00-2899-11ea-8bf2-ee6ac2216286.jpg)

![SNAG-0620](https://user-images.githubusercontent.com/46163555/71523946-43c03a00-2899-11ea-9f25-2f2b2c882ad3.jpg)


-----------------------------------

# Webcam Multi-Window Demo

In July 2020 a new demo was added that uses the new (released to GitHub only at this point) multi-window support.  This demo shows 3 video windows:

1. Your webcam's raw video stream
2. Grayscale version of the video
3. Fully colorized colored of the grayscale video

Here's a screenshot to give you a rough idea of what to expect from the demo.  The colors likely didn't do so well in this specific shot as there was a lot of background lighting.

![SNAG-0881](https://user-images.githubusercontent.com/46163555/88486988-9e189a80-cf4f-11ea-8dc7-727b7539bab9.jpg)


You will need to use the PySimpleGUI.py file from the project's GitHub http://www.PySimpleGUI.com.  Minimum version is 4.26.0.13.
