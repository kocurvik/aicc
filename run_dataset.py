import argparse
import os
import sys
import time

from res.video_params import get_video_filenames
from run_single_video_threaded import run_single_video_threaded


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=int, default=0, help='debug level')
    parser.add_argument('-fp', '--full-precision', action='store_true', default=False, help='disable automatic mixed precision and us fp32')
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def run(path, debug=0, full_precision=False):
    init_time = time.time()
    for vid_filename in get_video_filenames():
        vid_path = os.path.join(path, vid_filename)
        run_single_video_threaded(vid_path, debug=debug, full_precision=full_precision)

    print("Total time: {}".format(time.time() - init_time), file=sys.stderr)


if __name__ == '__main__':
    args = parse_args()
    run(args.path, args.debug, args.full_precision)


