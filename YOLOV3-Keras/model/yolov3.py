"""
    @Author: Tan.wt
    @Time: 2018/05/25
    @File: yolov3.py
    @License: Apache License
"""

import numpy as np
import keras.backend as K
from keras.models import load_model

class YOLO:
    def __init__(self, obj_threshold, nms_threshold):
        self.t1 = obj_threshold
        self.t2 = nms_threshold
        self._yolo = load_model('')

    def nms_boxes(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        '''

        :param boxes:
        :param scores:
        :return:
        '''
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        area = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            over = inter / (area[i] + area[order[1:]] - inter)
            indexs = np.where(over <= self.t2)[0]
            order = order[indexs + 1]

        keep = np.append(keep)

        return keep

