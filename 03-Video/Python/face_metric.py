### General imports ###
from __future__ import division

import numpy as np
import pandas as pd
import cv2
import datetime

from tqdm import tqdm
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


def get_video_metadata(video):
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    return length, width, height, fps


def process_video(parameters):
    first_time = datetime.datetime.now()
    shape_x = 48
    shape_y = 48
    emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    frame_index = 0
    display_frame = False
    metric_output = pd.DataFrame(columns=["Frame", "Faces", "Emotions"])

    model = load_model('Models/video.h5')
    face_detect = dlib.get_frontal_face_detector()

    video_capture = cv2.VideoCapture(parameters.input_video)
    length, width, height, fps = get_video_metadata(video_capture)
    print(length, width, height, fps)
    pbar = tqdm(total=length)

    while video_capture.isOpened():
        # Update progress bar
        pbar.update(1)
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_index += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)

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
            if len(rects) == 1:
                metric_output = metric_output.append({'Frame': frame_index,
                                                      'Faces': len(rects),
                                                      'Emotions': np.argmax(prediction[0, :])},
                                                     ignore_index=True)
            else:
                metric_output = metric_output.append({'Frame': frame_index,
                                                      'Faces': len(rects),
                                                      'Emotions': np.argmax(prediction[0, :][i])},
                                                     ignore_index=True)

        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

    later_time = datetime.datetime.now()
    print("Process time: ", later_time - first_time)
    metric_output.to_csv(parameters.output_metrics)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    return


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import pandas_bokeh

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def plot_result(parameters):
    emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    result = pd.read_csv(parameters.data)
    result = result.drop(["Unnamed: 0"], axis=1)

    """source = ColumnDataSource(result)
    plot = figure()
    result.plot_bokeh(kind="line", title="Title", figsize=(1000, 600), xlabel="frame", ylabel="ylabel")"""
    print([result["Emotions"].tolist().count(i) for i, emotion in enumerate(emotion_list)])
    print([truncate(result["Emotions"].tolist().count(i)/result.tail(1)["Frame"].item()*100, 2) for i, emotion in enumerate(emotion_list)])
    return


def main():
    parameters = parse_argument()


def parse_argument():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    process_video_parser = subparsers.add_parser('process_video')
    process_video_parser.add_argument('-i', '--input_video', help="Path input video")
    process_video_parser.add_argument('-o', '--output_metrics', default="metrics.csv", help="Path to the output metrics")
    process_video_parser.set_defaults(func=process_video)

    plot_result_parser = subparsers.add_parser('plot_result')
    plot_result_parser.add_argument('-d', '--data', help="Path to video metrics")
    plot_result_parser.set_defaults(func=plot_result)

    args = parser.parse_args()
    args.func(args)
    return args


if __name__ == "__main__":
    main()
