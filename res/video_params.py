import os
import sys

import cv2

VIDEO_PARAM_DICT = {'cam_1.mp4': [1, 1, 3000, 1280, 960], 'cam_1_dawn.mp4': [2, 1, 3000, 1280, 960],
                    'cam_1_rain.mp4': [3, 1, 2961, 1280, 960], 'cam_2.mp4': [4, 2, 18000, 1280, 720],
                    'cam_2_rain.mp4': [5, 2, 3000, 1280, 720], 'cam_3.mp4': [6, 3, 18000, 1280, 720],
                    'cam_3_rain.mp4': [7, 3, 3000, 1280, 720], 'cam_4.mp4': [8, 4, 27000, 1280, 960],
                    'cam_4_dawn.mp4': [9, 4, 4500, 1280, 960], 'cam_4_rain.mp4': [10, 4, 3000, 1280, 960],
                    'cam_5.mp4': [11, 5, 18000, 1280, 960], 'cam_5_dawn.mp4': [12, 5, 3000, 1280, 960],
                    'cam_5_rain.mp4': [13, 5, 3000, 1280, 960], 'cam_6.mp4': [14, 6, 18000, 1280, 960],
                    'cam_6_snow.mp4': [15, 6, 3000, 1280, 960], 'cam_7.mp4': [16, 7, 14400, 1280, 960],
                    'cam_7_dawn.mp4': [17, 7, 2400, 1280, 960], 'cam_7_rain.mp4': [18, 7, 3000, 1280, 960],
                    'cam_8.mp4': [19, 8, 3000, 1920, 1080], 'cam_9.mp4': [20, 9, 3000, 1920, 1080],
                    'cam_10.mp4': [21, 10, 2111, 1920, 1080], 'cam_11.mp4': [22, 11, 2111, 1920, 1080],
                    'cam_12.mp4': [23, 12, 1997, 1920, 1080], 'cam_13.mp4': [24, 13, 1966, 1920, 1080],
                    'cam_14.mp4': [25, 14, 3000, 2560, 1920], 'cam_15.mp4': [26, 15, 3000, 1920, 1080],
                    'cam_16.mp4': [27, 16, 3000, 1920, 1080], 'cam_17.mp4': [28, 17, 3000, 1920, 1080],
                    'cam_18.mp4': [29, 18, 3000, 1920, 1080], 'cam_19.mp4': [30, 19, 3000, 1920, 1080],
                    'cam_20.mp4': [31, 20, 3000, 1920, 1080]}


def get_video_params(path, check_frame_count=True):
    filename = os.path.basename(os.path.normpath(path))
    d = VIDEO_PARAM_DICT[filename]
    if check_frame_count:
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if d[2] != frame_count:
            print("Mismatch in saved frame count and cap frame count", file=sys.stderr)
            d[2] = frame_count

    return d


def get_video_filenames():
    return sorted(VIDEO_PARAM_DICT.keys())


def get_sorted_list(path=None):
    out = []
    for key, value in VIDEO_PARAM_DICT.items():
        line = [key]
        if path is not None:
            cap = cv2.VideoCapture(os.path.join(path, key))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if value[2] != frame_count:
                print("Mismatch in saved frame count and cap frame count", file=sys.stderr)
                value[2] = frame_count

        line.extend(value)
        out.append(line)
    return sorted(out, key=lambda item: item[3], reverse=True)

