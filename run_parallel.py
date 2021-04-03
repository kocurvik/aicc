import argparse
import os
from multiprocessing import Pool

import torch.multiprocessing as mp
mp.set_start_method('spawn')

from res.video_params import get_video_filenames
from threaded_pipeline import run_single_video_threaded


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=int, default=0, help='debug level')
    parser.add_argument('-p', '--pool_size', type=int, default=4, help='number of processes')
    parser.add_argument('path', help='path to the dataset folder')
    args = parser.parse_args()
    return args

def run(path, debug=0, pool_size=4):
    args =[(os.path.join(path, vid_filename), debug) for vid_filename in get_video_filenames()]

    with Pool(pool_size) as p:
        p.starmap(run_single_video_threaded, args)

if __name__ == '__main__':
    args = parse_args()
    run(args.path, args.debug, args.pool_size)