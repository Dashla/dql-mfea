#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# ------------------------------------------------------------
# Operations with models
# ------------------------------------------------------------
def analyse_model(model):
    shapes = []
    mask = []
    layer_dims = []
    for layer in model.layers:
        w = layer.get_weights()
        if len(w) == 0:
            mask.append(0)
        else:
            mask.append(1)
            layer_dims.append(layer.count_params())
            try:
                if s[1]:
                    shapes.append(s)
            except Exception as e:
                shapes.append([np.shape(w[i]) for i in range(len(w))])
    return shapes, mask, model.get_weights(), layer_dims


def flatten(weights):
    np_array = np.array([])
    for i in range(np.shape(weights)[0]):
        np_array = np.append(np_array, weights[i])
    return np_array


def unravel(weights, layer_shapes):
    reshaped_weights = []
    for layer_shape in layer_shapes:
        if type(layer_shape) == list:
            layer_weights = []
            for shape in layer_shape:
                to_r, weights = np.split(weights, [np.nanprod(shape)])
                layer_weights.append(np.reshape(to_r, shape))
            reshaped_weights.append(layer_weights)
        else:
            split = np.nanprod(layer_shape)
            to_r, weights = np.split(weights, [split])
            reshaped_weights.append(np.reshape(to_r, layer_shape))
    return reshaped_weights


def set_weights(model, weights, mask):
    for i, l in enumerate(model.layers):
        if mask[i] == 1:
            l.set_weights(weights.pop(0))


if __name__ == "__main__":
    pass
