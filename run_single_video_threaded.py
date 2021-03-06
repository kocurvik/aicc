import argparse
import sys
import time
from queue import Queue, Empty
from threading import Thread

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
    parser.add_argument('-n', '--new_thresh', type=float, default=0.4, help='threshold for new track')
    parser.add_argument('-t', '--track_thresh', type=float, default=0.2, help='threshold for new bbox for existing track')
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def video_reader_thread_fn(q_out, path):
    video_id, camera_id, max_frames, width, height = get_video_params(path)

    cap = cv2.VideoCapture(path)

    preprocess_function = get_img_transform(height, width, new_size=512)
    region_mask = get_region_mask(camera_id, height, width)
    region_mask = np.where(region_mask, 255, 0).astype(np.uint8)

    for i in range(max_frames):
        ret, frame = cap.read()
        frame = cv2.bitwise_and(frame, frame, mask=region_mask)
        img = preprocess_function(frame)

        q_out.put(img)


def model_thread_fn(q_in, q_out, path, full_precision=False):
    video_id, camera_id, max_frames, width, height = get_video_params(path)
    model = create_model()
    model = load_model(model, 'checkpoints/coco_tracking.pth')
    model.to(torch.device('cuda'))
    model.eval()

    pre_img = None

    for i in range(max_frames):
        img = q_in.get()

        img = torch.from_numpy(img).to(torch.device('cuda'))

        if pre_img is None:
            pre_img = img

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=not full_precision):
                out = model(img, pre_img, None)[-1]
                out = sigmoid_output(out)
                dets = generic_decode(out)
        pre_img = img

        q_out.put(dets)


def tracker_thread_fn(q_in, init_time, path, debug=0, new_thresh=0.4, track_thresh=0.2):
    video_id, camera_id, max_frames, width, height = get_video_params(path)

    postprocess_trans = get_postprocess_trans(height, width)
    tracker = Tracker(init_time, video_id, max_frames, camera_id, width, height, new_thresh=new_thresh, track_thresh=track_thresh)

    for i in range(max_frames):
        dets = q_in.get()
        get_time = time.time()

        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        dets = post_process(dets, postprocess_trans, track_thresh=track_thresh)[0]
        tracker.step(dets)

        if debug > 0 and i % 100 == 99:
            frame_time = time.time() - init_time
            FPS = (i + 1) / frame_time
            print("At frame {} FPS {}".format(i + 1, FPS), file=sys.stderr)

    tracker.finalize()

def run_single_video_threaded(path, debug=0, full_precision=False, new_thresh=0.4, track_thresh=0.2):
    init_time = time.time()
    if debug >= 1:
        print("Starting for video: {}".format(path), file=sys.stderr)

    q_in_model = Queue(10)
    q_out_model = Queue(10)

    reader_thread = Thread(target=video_reader_thread_fn, args=(q_in_model, path))
    model_thread = Thread(target=model_thread_fn, args=(q_in_model, q_out_model, path, full_precision))
    tracker_thread = Thread(target=tracker_thread_fn, args=(q_out_model, init_time, path, debug, new_thresh, track_thresh))

    reader_thread.start()
    model_thread.start()
    tracker_thread.start()

    reader_thread.join()
    model_thread.join()
    tracker_thread.join()

    if debug >= 1:
        print("Finished video: {}".format(path), file=sys.stderr)


if __name__ == '__main__':
    args = parse_args()
    run_single_video_threaded(args.path, args.debug, args.full_precision, new_thresh=args.new_thresh, track_thresh=args.track_thresh)