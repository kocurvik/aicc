from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from .networks.dla import DLASeg


def create_model():
    heads = {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}
    head_convs = {head: [256] for head in heads}
    model = DLASeg(34, heads=heads, head_convs=head_convs)
    return model


def load_model(model, model_path):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if (state_dict[k].shape != model_state_dict[k].shape):
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]

    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model