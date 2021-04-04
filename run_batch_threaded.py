import argparse
import os
import shutil
import sys
import tempfile
import time
from queue import Queue
from threading import Thread

import cv2
import torch
import numpy as np

from model.model import create_model, load_model
from model.postprocessing import get_postprocess_trans, post_process, sigmoid_output, generic_decode
from model.preprocessing import get_img_transform
from res.fields import get_region_mask
from res.video_params import get_sorted_list
from utils.tracker import Tracker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=int, default=0, help='debug level')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('-r', '--load_to_ram', action='store_true', default=False, help='load videos to RAM')
    parser.add_argument('path', help='path to the dataset folder')
    args = parser.parse_args()
    return args


class ReaderManager(object):
    def __init__(self, path, vid_list, load_to_ram):
        vid_filename = vid_list[0]
        video_path = os.path.join(path, vid_filename)

        if load_to_ram:
            tempdir = tempfile.TemporaryDirectory(dir='/dev/shm')
            copypath = os.path.join(tempdir.name, vid_filename)
            shutil.copy(video_path, copypath)
            video_path = copypath

        self.cap = cv2.VideoCapture(video_path)

        video_id, camera_id, max_frames, width, height = vid_list[1:]
        self.max_frames = max_frames

        self.preprocess_function = get_img_transform(height, width, new_size=512)

        region_mask = get_region_mask(camera_id, height, width)
        self.region_mask = np.where(region_mask, 255, 0).astype(np.uint8)

        self.prev_img = None
        self.n = 0

    def get_img(self):
        ret, frame = self.cap.read()
        frame = cv2.bitwise_and(frame, frame, mask=self.region_mask)
        img = self.preprocess_function(frame)
        img = torch.from_numpy(img).to(torch.device('cuda'))
        self.n += 1
        prev_img = self.prev_img if self.prev_img is not None else img
        self.prev_img = img

        return img, prev_img

    def is_done(self):
        return self.n >= self.max_frames

class TrackerManager(object):
    def __init__(self, vid_list, init_time):
        self.init_time = init_time

        video_id, camera_id, max_frames, width, height = vid_list[1:]
        self.max_frames = max_frames

        self.tracker = Tracker(init_time, video_id, max_frames, camera_id, width, height)

        self.postprocess_trans = get_postprocess_trans(height, width)

        self.prev_img = None
        self.n = 0

    def process_output(self, dets):
        dets = post_process(dets, self.postprocess_trans)[0]
        self.tracker.step(dets)

    def is_done(self):
        return self.n >= self.max_frames


def reader_thread_fn(q_out, q_times, batch_size, path, multi_vid_list, load_to_ram=False):
    vid_managers = []
    for vid_list in multi_vid_list[:batch_size]:
        q_times.put(time.time())
        vid_managers.append(ReaderManager(path, vid_list, load_to_ram))

    next_video_id = len(vid_managers)
    while len(vid_managers) > 0:
        imgs = [manager.get_img() for manager in vid_managers]
        cur_imgs = torch.cat([x[0] for x in imgs], dim=0)
        prev_imgs = torch.cat([x[1] for x in imgs], dim=0)
        q_out.put([cur_imgs, prev_imgs])

        for i, manager in reversed(list(enumerate(vid_managers))):
            if manager.is_done():
                if next_video_id < len(multi_vid_list):
                    q_times.put(time.time())
                    vid_managers[i] = ReaderManager(path, multi_vid_list[next_video_id])
                    next_video_id += 1
                else:
                    del vid_managers[i]
    q_out.put([None, None])


def model_thread_fn(q_in, q_out):
    model = create_model()
    model = load_model(model, 'checkpoints/coco_tracking.pth')
    model.to(torch.device('cuda'))
    model.eval()

    while True:
        cur_imgs, prev_imgs = q_in.get(timeout=30)
        if cur_imgs is None:
            break
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                out = model(cur_imgs, prev_imgs, None)[-1]
                out = sigmoid_output(out)
                dets = generic_decode(out)
        q_out.put(dets)


def tracker_thread_fn(q_in, q_times, multi_vid_list, model_loading_time, batch_size, debug=0):
    tracker_managers = []
    thread_init_time = None
    for vid_list in multi_vid_list[:batch_size]:
        init_time = q_times.get()
        if thread_init_time is None:
            thread_init_time = init_time - model_loading_time
        tracker_managers.append(TrackerManager(vid_list, init_time - model_loading_time))

    next_video_id = len(tracker_managers)


    processed_frames = 0

    while len(tracker_managers) > 0:
        dets = q_in.get()
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        processed_frames += len(tracker_managers)

        for i, manager in reversed(list(enumerate(tracker_managers))):
            single_dets = {k: v[np.newaxis, i] for k, v in dets.items()}
            manager.process_output(single_dets)

            if manager.is_done():
                if next_video_id < len(multi_vid_list):
                    init_time = q_times.get()
                    tracker_managers[i] = TrackerManager(multi_vid_list[next_video_id], init_time - model_loading_time)
                    next_video_id += 1
                else:
                    del tracker_managers[i]
        if debug:
            FPS = processed_frames / (time.time() - thread_init_time)
            print("At frame {} FPS {}".format(processed_frames, FPS), file=sys.stderr)


def run(path, batch_size=4, debug=0, load_to_ram = False):
    pre_model_load = time.time()

    model = create_model()
    model = load_model(model, 'checkpoints/coco_tracking.pth')
    model.to(torch.device('cuda'))
    model.eval()
    del model

    model_loading_time = time.time() - pre_model_load
    multi_vid_list = get_sorted_list()

    q_in_model = Queue(10)
    q_out_model = Queue(10)
    q_times = Queue(batch_size + 10)

    reader_thread = Thread(target=reader_thread_fn, args=(q_in_model, q_times, batch_size, path, multi_vid_list, load_to_ram))
    model_thread = Thread(target=model_thread_fn, args=(q_in_model, q_out_model))
    tracker_thread = Thread(target=tracker_thread_fn, args=(q_out_model, q_times, multi_vid_list, model_loading_time, batch_size, debug))

    reader_thread.start()
    model_thread.start()
    tracker_thread.start()

    reader_thread.join()
    model_thread.join()
    tracker_thread.join()

    print("Total time: {}".format(time.time() - pre_model_load), file=sys.stderr)


if __name__ == '__main__':
    args = parse_args()
    run(args.path, args.batch_size, args.debug, args.load_to_ram)