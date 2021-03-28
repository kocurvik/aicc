import time
import copy
import os

import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
# from sklearn.utils.linear_assignment_ import linear_assignment
from fields.interest import get_mask_movements_heatmaps

import matplotlib
# matplotlib.use('TkAgg')

if os.name == 'nt':
    matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

class Track(object):
    def __init__(self, tracking_id, frame_cnt, item):
        self.tracking_id = tracking_id
        self.frames = [frame_cnt]
        item['tracking_id'] = tracking_id
        self.items = [item]
        self.frame = None

    def assign(self, frame_cnt, item):
        self.frames.append(frame_cnt)
        item['tracking_id'] = self.tracking_id
        self.items.append(item)

    def last(self):
        return self.items[-1]

    def is_alive(self, frame_count):
        if frame_count - self.frames[-1] < 5:
            return True


class Tracker(object):
    def __init__(self, opt, init_time, vid_id, max_frames, camera_id, width, height):
        self.opt = opt
        self.init_time = init_time
        self.vid_id = vid_id
        self.width = width
        self.height = height
        self.max_frames = max_frames

        self.movements, self.corners, self.distance_heatmaps, self.proportion_heatmaps = get_mask_movements_heatmaps(camera_id, height, width)

        # for i in range(len(self.distance_heatmaps)):
        #     cv2.imshow("distance", self.distance_heatmaps[i]/ np.max(self.distance_heatmaps[i]))
        #     cv2.imshow("proportion", self.proportion_heatmaps[i])
        #     cv2.waitKey(0)

        self.id_count = 0
        self.frame_count = 0
        self.tracks = []

    def reset(self):
        self.id_count = 0
        self.frame_count = 0
        self.tracks = []

    def filter_results(self, results):
        results_cars = [item for item in results if item['class'] == 3]
        results_trucks = [item for item in results if item['class'] == 8]
        results_buses = [item for item in results if item['class'] == 6]

        # NMS buses -> trucks
        bbox_cars = np.array([item['bbox'] for item in results_cars]).reshape(-1, 4)
        bbox_trucks = np.array([item['bbox'] for item in results_trucks]).reshape(-1, 4)
        bbox_buses = np.array([item['bbox'] for item in results_buses]).reshape(-1, 4)

        if len(results_buses) > 0:
            iou_buses_trucks = iou(bbox_trucks, bbox_buses)
            good_buses = np.all(iou_buses_trucks < 0.7, axis=0)

            results_trucks.extend([results_buses[i] for i in range(len(results_buses)) if good_buses[i]])
            bbox_trucks = np.row_stack([bbox_trucks, bbox_buses[good_buses, :]])

        # NMS trucks -> cars
        if len(results_trucks) > 0:
            iou_cars_trucks = iou(bbox_cars, bbox_trucks)
            good_trucks = np.all(iou_cars_trucks < 0.7, axis=0)
            results_cars.extend([results_trucks[i] for i in range(len(results_trucks)) if good_trucks[i]])

        return results_cars

    def add_sizes(self, results):
        for item in results:
            item['size'] = ((item['bbox'][2] - item['bbox'][0]) * (item['bbox'][3] - item['bbox'][1]))

        return results

    def debug_track(self, pos_x, pos_y, corner_x, corner_y, color=(0, 255, 0)):
        vis = np.copy(self.frame)
        for i in range(len(pos_x)):
            vis = cv2.circle(vis, (pos_x[i], pos_y[i]), 5, color=color)
            vis = cv2.circle(vis, (corner_x[i], corner_y[i]), 3, color=color, thickness=-1)

        cv2.imshow("Track debug", vis)
        cv2.waitKey(0)

    def generate_entry(self, track):
        if len(track.frames) < 5:
            return

        positions = np.array([item['ct'] for item in track.items]).astype(np.int32)
        positions[:, 0] = np.clip(positions[:, 0], 0, self.width - 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, self.height - 1)

        distances = self.distance_heatmaps[:, positions[:, 1], positions[:, 0]]
        proportions = self.proportion_heatmaps[:, positions[:, 1], positions[:, 0]]
        mean_distances = np.mean(distances, axis=-1)
        mean_distances += 1e18 * (proportions[:, 0] > proportions[:, -1])
        std_distances = np.std(distances, axis=-1)
        path = np.argmin(mean_distances + 3 * std_distances)

        corner_positiots_x = np.array([item['bbox'][self.corners[path, 0]] for item in track.items], dtype=np.int32)
        corner_positiots_y = np.array([item['bbox'][self.corners[path, 1]] for item in track.items], dtype=np.int32)

        corner_positiots_x = np.clip(corner_positiots_x, 0, self.width - 1)
        corner_positiots_y = np.clip(corner_positiots_y, 0, self.height - 1)

        proportions = self.proportion_heatmaps[path, corner_positiots_y, corner_positiots_x]

        if np.max(proportions) < 0.6 or np.max(proportions) - np.min(proportions) < 0.25 * min(self.frame_count / 50, 1):
            if self.opt.debug > 0:
                self.debug_track(positions[:, 0], positions[:, 1], corner_positiots_x, corner_positiots_y, color=(0, 0, 255))
            return

        times = np.array(track.frames)
        weights = ((times - times[0]) / (times[-1] - times[0])) ** 1.5
        regr = LinearRegression()
        # regr.fit(times[:, np.newaxis], proportions, sample_weight=weights)
        # start_range = (len(times) * 3) // 5
        # end_range = (len(times) * 9) // 10
        regr.fit(times[-10: -3].reshape(-1, 1), proportions[-10: -3].reshape(-1, 1))

        if self.opt.debug > 1:
            plt.plot(times, proportions)
            plt.plot(times[:, np.newaxis], regr.predict(times[:, np.newaxis]))
            plt.show()

        if regr.coef_ <= 0.0:
            return

        projected_last_frame = ((1 - regr.intercept_) / regr.coef_)[0]

        if projected_last_frame > self.max_frames:
            return

        truck_num = sum([item['class'] == 6 or item['class'] == 8 for item in track.items])
        cls = 2 if truck_num / len(track.frames) > 0.5 else 1
        gen_time = time.time() - self.init_time
        print('{} {} {} {} {}'.format(gen_time, self.vid_id, np.int32(projected_last_frame[0]), path + 1, cls))

        if self.opt.debug > 0:
            self.debug_track(positions[:, 0], positions[:, 1], corner_positiots_x, corner_positiots_y)

    def step(self, results, public_det=None):
        self.frame_count += 1

        results = self.filter_results(results)
        results = self.add_sizes(results)

        # item_size = np.array([item['size'] for item in results])
        # track_size = np.array([track.last()['size'] for track in self.tracks])
        # dets = np.array([det['ct'] + det['tracking'] for det in results], np.float32)  # N x 2
        # tracks = np.array([track.last()['ct'] for track in self.tracks], np.float32)  # M x 2
        # dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
        # invalid = ((dist > track_size.reshape(1, M)) + (dist > item_size.reshape(N, 1))) > 0
        # dist = dist + invalid * 1e18

        track_bboxes = np.array([track.last()['bbox'] for track in self.tracks]).reshape(-1, 4)
        det_bboxes = np.array([item['bbox'] for item in results]).reshape(-1, 4)
        det_bboxes += np.tile(np.array([item['tracking'] for item in results]), (1, 2)).reshape(-1, 4)

        ious = iou(track_bboxes, det_bboxes)
        # print(ious)

        matches = []
        unmatched_dets = []
        unmatched_tracks = np.ones(len(self.tracks), dtype=bool)

        if len(self.tracks) == 0:
            unmatched_dets = [i for i in range(len(results))]
        else:
            for j in range(len(results)):
                i = np.argmax(ious[:, j])
                if ious[i, j] > 0.1:
                    matches.append([i, j])
                    ious[i, :] = 0.0
                    unmatched_tracks[i] = False
                else:
                    unmatched_dets.append(j)

        for m in matches:
            self.tracks[m[0]].assign(self.frame_count, results[m[1]])

        for i in reversed(unmatched_dets):
            item = results[i]
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                track = Track(self.id_count, self.frame_count, item)
                self.tracks.append(track)

        for i, val in reversed(list(enumerate(unmatched_tracks))):
            if val:
                track = self.tracks[i]
                if not track.is_alive(self.frame_count):
                    self.generate_entry(track)
                    del self.tracks[i]

        ret = [track.last() for track in self.tracks if track.is_alive(self.frame_count)]
        return ret


def iou(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.array([[]]).reshape(len(A), len(B))

    intersections = (np.maximum(0, np.minimum(A[:, np.newaxis, 2:], B[:, 2:]) - np.maximum(A[:, np.newaxis, :2], B[:, :2]))).prod(-1)
    unions = (A[:, np.newaxis, 2:] - A[:, np.newaxis, :2]).prod(-1) + (B[:, 2:] - B[:, :2]).prod(-1) - intersections
    return intersections / unions


# def iou(bbox1, bbox2):
#     x_left = max(bbox1[0], bbox2[0])
#     y_top = max(bbox1[1], bbox2[1])
#     x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
#     y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
#
#     if x_right < x_left or y_bottom < y_top:
#         return 0.0
#
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)
#     last_area = bbox1[2] * bbox1[3]
#     query_area = bbox2[2] * bbox2[3]
#
#     return intersection_area / float(last_area + query_area - intersection_area)


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] > 0.5:
            dist[:, j] = 0
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)
