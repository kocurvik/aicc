import datetime
import time
import os
import sys

import cv2
import numpy as np

from detector import Detector
from opts import opts
from fields.interest import get_mask_movements_heatmaps, get_region_mask
from utils.tracker import Tracker

import matplotlib
if os.name == 'nt':
    matplotlib.use('Qt5Agg')

def get_img_transform(height, width, new_size=512):
    ratio = float(new_size) / max([height, width])

    shift = ratio * np.abs(height - width) / 2

    if width > height:
        A = np.array([[ratio, 0, 0], [0, ratio, shift]])
    else:
        A = np.array([[ratio, 0, shift], [0, ratio, 0]])

    mean = np.array([[[0.40789654, 0.44719302, 0.47026115]]], dtype=np.float32)
    std = np.array([[[0.28863828, 0.27408164, 0.27809835]]], dtype=np.float32)

    def _preprocess_f(img):
        img = cv2.warpAffine(img, A, (new_size, new_size))
        img = ((img / 255.0 - mean) / std).astype(np.float32)
        img = img.transpose(2, 0, 1).reshape(1, 3, new_size, new_size)
        return img

    return _preprocess_f


def run(opt):
    # opt.debug = max(opt.debug, 1)
    init_time = time.time()
    processed_frames = 0
    total_frames = 185446

    with open(os.path.join(opt.demo, 'list_video_id.txt'), 'r') as f:
        lines = f.read().splitlines()

    with open(os.path.join(opt.demo, 'datasetA_vid_stats.txt'), 'r') as f:
        stats_lines = f.read().splitlines()

    max_frames_list = [int(line.split('\t')[2]) for line in stats_lines[1:]]

    detector = Detector(opt, None)

    for line, max_frames in zip(lines, max_frames_list):
        vid_id = line.split(' ')[0]
        vid_filename = line.split(' ')[1]
        camera_label = int(vid_filename.split('.')[0].split('_')[1])

        vid_path = os.path.join(opt.demo, vid_filename)
        print("vid_path: {}".format(vid_path), file=sys.stderr)

        cap = cv2.VideoCapture(vid_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Vid loaded with width {} height {}".format(width, height), file=sys.stderr)

        region_mask = get_region_mask(camera_label, height, width)
        region_mask = np.where(region_mask, 255, 0).astype(np.uint8)

        tracker = Tracker(opt, init_time, vid_id, max_frames, camera_label, width, height)
        detector.tracker = tracker

        ret = True
        # cnt = 0
        # results = {}

        while ret:
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.bitwise_and(img, img, mask=region_mask)
            ret = detector.run(img)
            processed_frames += 1

            if processed_frames % 100 == 0:
                remaining_seconds = (total_frames - processed_frames) * (time.time() - init_time) / processed_frames
                print("Frame {}/{} ETA {}".format(processed_frames, total_frames, datetime.timedelta(seconds=(remaining_seconds))), file=sys.stderr)
            # cnt += 1
            # results[cnt] = ret['results']

            if opt.debug > 0:
                cv2.waitKey(0)



if __name__ == '__main__':
    opt = opts().init()
    run(opt)