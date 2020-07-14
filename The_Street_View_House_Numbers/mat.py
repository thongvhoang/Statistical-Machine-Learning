#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:43:48 2020

@author: tintin
"""
import h5py
def get_name(f, index=0):
    name = f['/digitStruct/name']
    return ''.join([chr(v[0]) for v in f[(name[index][0])]])


def get_bbox(f, index=0):
     meta = { key : [] for key in ['height', 'left', 'top', 'width', 'label']}
     def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(int(obj[0][0]))
        else:
            for k in range(obj.shape[0]):
                vals.append(int(f[obj[k][0]][0][0]))
        meta[name] = vals
     box = f['/digitStruct/bbox'][index]
     f[box[0]].visititems(print_attrs)
     return meta
train_name = []
train_bbox = []
size = 100
with h5py.File('/home/tintin/Study/Data-Science/Statistical-Machine-Learning/The_Street_View_House_Numbers/Datasets/train_digitStruct.mat', 'r') as train_data:
    
    # Getting data
    for i in range(size):
        train_name.append(get_name(train_data, i))
        train_bbox.append(get_bbox(train_data, i))

        # Cropping current image
        cropped = []
        for k in range(len(train_bbox[i]['top'])):
            top = train_bbox[i]['top'][k]
            left = train_bbox[i]['left'][k]
            height = train_bbox[i]['height'][k]
            width = train_bbox[i]['width'][k]
            cropped.append(images[i][top:top+height,left:left+width])
        images[i] = cropped

import cv2
data = []    
image = cv2.imread("/home/tintin/Study/Data-Science/Statistical-Machine-Learning/The_Street_View_House_Numbers/Datasets/train/train/1.png")
data.append(image)
