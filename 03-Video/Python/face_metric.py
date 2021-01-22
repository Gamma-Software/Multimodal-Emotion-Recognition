### General imports ###
from __future__ import division

import numpy as np
import pandas as pd
import cv2
import datetime

from time import time
from time import sleep
import re
import os

import argparse
from collections import OrderedDict

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage

import dlib

from tensorflow.keras.models import load_model
from imutils import face_utils

import requests

global shape_x
global shape_y
global input_shape
global nClasses

def show_webcam(parameters):
    first_time = datetime.datetime.now()
    shape_x = 48
    shape_y = 48
    emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    frame_index = 0
    display_frame = False
    metric_output = pd.DataFrame(columns=["Faces", "Emotions"])

    model = load_model('Models/video.h5')
    face_detect = dlib.get_frontal_face_detector()

    video_capture = cv2.VideoCapture(parameters.input_video)

    while video_capture.isOpened():
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_index += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)

        emotions = []
        for (i, rect) in enumerate(rects):
            # Identify face coordinates
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y + h, x:x + w]

            # Zoom on extracted face
            try:
                face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
            except ZeroDivisionError:
                print("Avoid division by zero")
                break

            # Cast type float
            face = face.astype(np.float32)

            # Scale
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))

            # Make Prediction
            prediction = model.predict(face)
            emotions = np.append(emotions, emotion_list[np.argmax(prediction[0, :][i])])

            # TODO sauvegarder la photo zoomee

        metric_output = metric_output.append({'Faces': len(rects),
                                              'Emotions': emotions},
                                             ignore_index=True)
        print(metric_output.tail(1))

        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

    later_time = datetime.datetime.now()
    print("Process time: ", later_time - first_time)
    metric_output.to_csv(parameters.output_metrics)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    parameters = parse_argument()
    show_webcam(parameters)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video', help="Path input video")
    parser.add_argument('-o', '--output_metrics', default="metrics.csv", help="Path to the output metrics")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
