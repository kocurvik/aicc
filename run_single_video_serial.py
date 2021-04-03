import argparse
import sys
import time

import cv2
import torch
import numpy as np

from model.model import create_model, load_model
from model.postprocessing import generic_decode, sigmoid_output, get_postprocess_trans, post_process
from model.preprocessing import get_img_transform
from res.fields import get_region_mask
from res.video_params import get_video_params
from utils.tracker import Tracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=int, default=0, help='debug level')
    parser.add_argument('-fp', '--full-precision', action='store_true', default=False, help='disable automatic mixed precision and us fp32')
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def run_single_video_serial(path, debug=0, full_precision=False):
    init_time = time.time()
    if debug >= 1:
        print("Starting for video: {}".format(path), file=sys.stderr)

    video_id, camera_id, max_frames, width, height = get_video_params(path)

    cap = cv2.VideoCapture(path)

    model = create_model()
    model = load_model(model, 'checkpoints/coco_tracking.pth')
    model.to(torch.device('cuda'))
    model.eval()

    tracker = Tracker(init_time, video_id, max_frames, camera_id, width, height)

    preprocess_function = get_img_transform(height, width, new_size=512)
    postprocess_trans = get_postprocess_trans(height, width)
    region_mask = get_region_mask(camera_id, height, width)
    region_mask = np.where(region_mask, 255, 0).astype(np.uint8)

    pre_img = None

    for i in range(max_frames):
        ret, frame = cap.read()
        frame = cv2.bitwise_and(frame, frame, mask=region_mask)
        if debug >= 2:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            tracker.frame = frame

        img = preprocess_function(frame)
        img = torch.from_numpy(img).to(torch.device('cuda'))

        if pre_img is None:
            pre_img = img

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=not full_precision):
                out = model(img, pre_img, None)[-1]
                out = sigmoid_output(out)
                dets = generic_decode(out)

        pre_img = img

        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        dets = post_process(dets, postprocess_trans)[0]
        tracker.step(dets)

        if debug >= 1 and i % 100 == 99:
            frame_time = time.time() - init_time
            FPS = (i + 1) / frame_time
            print("At frame {} FPS {}".format(i + 1, FPS), file=sys.stderr)

    if debug >= 1:
        print("Finished video: {}".format(path), file=sys.stderr)

if __name__ == '__main__':
    args = parse_args()
    run_single_video_serial(args.path, args.debug, args.full_precision)