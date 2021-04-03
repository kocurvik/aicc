import copy
import math

import numpy as np
import cv2
import torch
from utils.image import gaussian_radius, draw_umich_gaussian, affine_transform, get_affine_transform


def get_img_transform(height, width, new_size=512):
    # ratio = float(new_size) / max([height, width])
    # shift = ratio * np.abs(height - width) / 2
    # if width > height:
    #     A = np.array([[ratio, 0, 0], [0, ratio, shift]])
    # else:
    #     A = np.array([[ratio, 0, shift], [0, ratio, 0]])

    mean = np.array([[[0.40789654, 0.44719302, 0.47026115]]], dtype=np.float32)
    std = np.array([[[0.28863828, 0.27408164, 0.27809835]]], dtype=np.float32)

    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [new_size, new_size])

    # def _preprocess_f(img):
    #     img = cv2.warpAffine(img, A, (new_size, new_size))
    #     img = ((img / 255.0 - mean) / std).astype(np.float32)
    #     img = img.transpose(2, 0, 1).reshape(1, 3, new_size, new_size)
    #     return img

    def _preprocess_f(img):
        img = cv2.warpAffine(img, trans_input, (new_size, new_size), flags=cv2.INTER_LINEAR)
        img = ((img / 255. - mean) / std).astype(np.float32)

        img = img.transpose(2, 0, 1).reshape(1, 3, new_size, new_size)
        return img

    return _preprocess_f


def trans_bbox(self, bbox, trans, width, height):
    '''
    Transform bounding boxes according to image crop.
    '''
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox


def get_additional_inputs(dets, meta, with_hm=True):
    '''
    Render input heatmap from previous trackings.
    '''
    trans_input, trans_output = meta['trans_input'], meta['trans_output']
    inp_width, inp_height = meta['inp_width'], meta['inp_height']
    out_width, out_height = meta['out_width'], meta['out_height']
    input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

    output_inds = []
    for det in dets:
        if det['score'] < 0.2 or det['active'] == 0:
            continue
        bbox = trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
        bbox_out = trans_bbox(det['bbox'], trans_output, out_width, out_height)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if (h > 0 and w > 0):
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            if with_hm:
                draw_umich_gaussian(input_hm[0], ct_int, radius)
            ct_out = np.array(
                [(bbox_out[0] + bbox_out[2]) / 2,
                 (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
            output_inds.append(ct_out[1] * out_width + ct_out[0])
    if with_hm:
        input_hm = input_hm[np.newaxis]
        input_hm = torch.from_numpy(input_hm).to(torch.device('cuda'))
    output_inds = np.array(output_inds, np.int64).reshape(1, -1)
    output_inds = torch.from_numpy(output_inds).to(torch.device('cuda'))
    return input_hm, output_inds