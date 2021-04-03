import argparse
import os
import sys
import time

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
    parser.add_argument('path', help='path to the dataset folder')
    args = parser.parse_args()
    return args


class VideoManager(object):
    def __init__(self, path, vid_list, model_loading_time):
        init_time = time.time() - model_loading_time
        self.init_time = init_time

        vid_filename = vid_list[0]
        video_path = os.path.join(path, vid_filename)
        self.cap = cv2.VideoCapture(video_path)

        video_id, camera_id, max_frames, width, height = vid_list[1:]
        self.max_frames = max_frames

        self.tracker = Tracker(init_time, video_id, max_frames, camera_id, width, height)

        self.preprocess_function = get_img_transform(height, width, new_size=512)
        self.postprocess_trans = get_postprocess_trans(height, width)

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

    def process_output(self, dets):
        dets = post_process(dets, self.postprocess_trans)[0]
        self.tracker.step(dets)

    def is_done(self):
        return self.n >= self.max_frames


def run(path, batch_size=4, debug=0):
    pre_model_load = time.time()

    model = create_model()
    model = load_model(model, 'checkpoints/coco_tracking.pth')
    model.to(torch.device('cuda'))
    model.eval()

    model_loading_time = time.time() - pre_model_load

    multi_vid_list = get_sorted_list()
    vid_managers = [VideoManager(path, vid_list, model_loading_time) for vid_list in multi_vid_list[:batch_size]]

    next_video_id = len(vid_managers)
    processed_frames = 0

    done = False
    while len(vid_managers) > 0:
        imgs = [manager.get_img() for manager in vid_managers]
        cur_imgs = torch.cat([x[0] for x in imgs], dim=0)
        prev_imgs = torch.cat([x[1] for x in imgs], dim=0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                out = model(cur_imgs, prev_imgs, None)[-1]
                out = sigmoid_output(out)
                dets = generic_decode(out)

        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        processed_frames += len(vid_managers)

        for i, manager in reversed(list(enumerate(vid_managers))):
            single_dets = {k: v[np.newaxis, i] for k, v in dets.items()}
            manager.process_output(single_dets)

            if manager.is_done():
                if next_video_id < len(multi_vid_list):
                    vid_managers[i] = VideoManager(path, multi_vid_list[next_video_id], model_loading_time)
                    next_video_id += 1
                else:
                    del vid_managers[i]

        if debug:
            frame_time = time.time() - pre_model_load
            FPS = processed_frames / frame_time
            print("At frame {} FPS {}".format(processed_frames, FPS), file=sys.stderr)


if __name__ == '__main__':
    args = parse_args()
    run(args.path, args.batch_size, args.debug)