import os
import json

import numpy as np
from PIL import Image, ImageDraw

TRACK_DIR = os.path.join(os.path.dirname(__file__), 'tracks')
REGION_DIR = os.path.join(os.path.dirname(__file__), 'regions')


def get_mask_movements_heatmaps(camera_id, height, width):
    with open(os.path.join(TRACK_DIR, 'cam_%d.json' % (camera_id))) as f:
        data = json.load(f)

    movements = {int(shape['label']): np.array(shape['points']) for shape in data['shapes']}
    assert len(movements) == max(movements.keys())
    distance_heatmaps, proportion_heatmaps = get_movement_heatmaps(movements, height, width)

    last_dirs = np.zeros([len(movements), 2])
    for i in range(len(movements)):
        last_dirs[i] = movements[i + 1][-1] - movements[i + 1][-2]

    corners = np.zeros_like(last_dirs)
    for i in range(last_dirs.shape[0]):
        if last_dirs[i, 0] < 0 and last_dirs[i, 1] < 0:
            corners[i] = np.array([2, 3])
        elif last_dirs[i, 0] > 0 and last_dirs[i, 1] < 0:
            corners[i] = np.array([0, 3])
        elif last_dirs[i, 0] > 0 and last_dirs[i, 1] > 0:
            corners[i] = np.array([0, 1])
        else:
            corners[i] = np.array([2, 1])

    return movements, corners.astype(np.int32), distance_heatmaps, proportion_heatmaps


def get_region_mask(camera_id, height, width):
    region = np.loadtxt(os.path.join(REGION_DIR, 'cam_%d.txt' % (camera_id)), delimiter=',', dtype=np.int)
    img = Image.new('L', (width, height), 0)
    region = region.flatten().tolist()
    ImageDraw.Draw(img).polygon(region, outline=0, fill=255)
    mask = np.array(img).astype(np.bool)
    return mask


def get_movement_heatmaps(movements, height, width):
    distance_heatmaps = np.empty((len(movements), height, width))
    proportion_heatmaps = np.empty((len(movements), height, width))
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack([xs.flatten(), ys.flatten()], axis=1)
    for label, movement_vertices in movements.items():
        vectors = movement_vertices[1:] - movement_vertices[:-1]
        lengths = np.linalg.norm(vectors, axis=-1) + 1e-4
        rel_lengths = lengths / lengths.sum()
        vertex_proportions = np.cumsum(rel_lengths)
        vertex_proportions = np.concatenate([[0], vertex_proportions[:-1]])
        offsets = ((points[:, None] - movement_vertices[None, :-1]) * vectors[None]).sum(axis=2)
        fractions = np.clip(offsets / (lengths ** 2), 0, 1)
        targets = movement_vertices[:-1] + fractions[:, :, None] * vectors
        distances = np.linalg.norm(points[:, None] - targets, axis=2)
        nearest_segment_ids = distances.argmin(axis=1)
        nearest_segment_fractions = fractions[np.arange(fractions.shape[0]), nearest_segment_ids]
        distance_heatmap = distances.min(axis=1)
        proportion_heatmap = vertex_proportions[nearest_segment_ids] + \
                             rel_lengths[nearest_segment_ids] * nearest_segment_fractions
        distance_heatmaps[label - 1, ys, xs] = distance_heatmap.reshape(height, width)
        proportion_heatmaps[label - 1, ys, xs] = proportion_heatmap.reshape(height, width)
    return distance_heatmaps, proportion_heatmaps
