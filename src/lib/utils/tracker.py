import time
import copy

import numpy as np
import cv2
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.utils.linear_assignment_ import linear_assignment
from fields.interest import get_mask_movements_heatmaps

import matplotlib
# matplotlib.use('TkAgg')
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
    def __init__(self, opt, init_time, vid_id, camera_id, width, height):
        self.opt = opt
        self.init_time = init_time
        self.vid_id = vid_id
        self.width = width
        self.height = height

        self.movements, self.corners,self.distance_heatmaps, self.proportion_heatmaps = get_mask_movements_heatmaps(camera_id, height, width)

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

        results_trucks_buses = []
        for item_bus in results_buses:
            keep = True
            for item_truck in results_trucks:
                if iou(item_truck['bbox'], item_bus['bbox']) > 0.8:
                    keep = False
                    break
            if keep:
                results_trucks_buses.append(item_bus)
        results_trucks_buses.extend(results_trucks)

        # NMS trucks -> cars
        results = []
        for item_truck in results_trucks_buses:
            keep = True
            for item_car in results_cars:
                if iou(item_car['bbox'], item_truck['bbox']) > 0.8:
                    keep = False
                    break
            if keep:
                results.append(item_truck)
        results.extend(results_cars)

        return results

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

        proportions = self.proportion_heatmaps[path, corner_positiots_y, corner_positiots_x]

        if np.max(proportions) < 0.6 or np.max(proportions) - np.min(proportions) < 0.25:
            if self.opt.debug > 0:
                self.debug_track(positions[:, 0], positions[:, 1], corner_positiots_x, corner_positiots_y, color=(0, 0, 255))
            return

        times = np.array(track.frames)
        weights = ((times - times[0]) / (times[-1] - times[0])) ** 1.5
        regr = LinearRegression()
        # regr.fit(times[:, np.newaxis], proportions, sample_weight=weights)
        regr.fit(times[-5:, np.newaxis], proportions[-5:])

        if self.opt.debug > 1:
            plt.plot(times, proportions)
            plt.plot(times[:, np.newaxis], regr.predict(times[:, np.newaxis]))
            plt.show()

        if regr.coef_ < 0.0:
            return

        projected_last_frame = (1 - regr.intercept_) / regr.coef_

        truck_num = sum([item['class'] == 6 or item['class'] == 8 for item in track.items])
        cls = 2 if truck_num / len(track.frames) > 0.6 else 1
        gen_time = time.time() - self.init_time
        print('{} {} {} {} {}'.format(gen_time, self.vid_id, np.int32(projected_last_frame[0]), path + 1, cls))
        if self.opt.debug > 0:
            self.debug_track(positions[:, 0], positions[:, 1], corner_positiots_x, corner_positiots_y)

    def step(self, results, public_det=None):
        self.frame_count += 1

        results = self.filter_results(results)
        results = self.add_sizes(results)
        N = len(results)
        M = len(self.tracks)

        item_size = np.array([item['size'] for item in results])
        track_size = np.array([track.last()['size'] for track in self.tracks])

        dets = np.array([det['ct'] + det['tracking'] for det in results], np.float32)  # N x 2
        tracks = np.array([track.last()['ct'] for track in self.tracks], np.float32)  # M x 2
        dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M

        invalid = ((dist > track_size.reshape(1, M)) + (dist > item_size.reshape(N, 1))) > 0
        dist = dist + invalid * 1e18

        if self.opt.hungarian:
            item_score = np.array([item['score'] for item in results], np.float32)  # N
            dist[dist > 1e18] = 1e18
            matched_indices = linear_assignment(dist)
        else:
            matched_indices = greedy_assignment(copy.deepcopy(dist))
        unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]

        if self.opt.hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        ret = []
        for m in matches:
            track = results[m[0]]
            self.tracks[m[1]].assign(self.frame_count, track)

        for i in unmatched_dets:
            item = results[i]
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                track = Track(self.id_count, self.frame_count, item)
                self.tracks.append(track)

        for i in reversed(unmatched_tracks):
            track = self.tracks[i]
            if not track.is_alive(self.frame_count):
                self.generate_entry(track)
                del self.tracks[i]

        ret = [track.last() for track in self.tracks if track.is_alive(self.frame_count)]

        return ret


def iou(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    last_area = bbox1[2] * bbox1[3]
    query_area = bbox2[2] * bbox2[3]

    return intersection_area / float(last_area + query_area - intersection_area)


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)
