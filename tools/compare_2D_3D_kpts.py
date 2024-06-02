# matplotlib
# numpy
import numpy as np
import random
from PIL import Image
# misc
import time
from datetime import datetime
# from torchinfo import summary
import cv2

import pandas as pd
import matplotlib.pyplot as plt

# logging
import os
import logging

import sys
import io

import argparse
import platform
# import mediapipe
import mediapipe as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network.PIXIE.utils.compute_body_vertices import ComputeBodyVerticesKpts
from utils.detect_2d_kpts import Detect2DKptsMediaPipe

def main_webcam():
    # 创建视频捕捉对象
    cap = cv2.VideoCapture(0)
    detector = Detect2DKptsMediaPipe()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 获取关键点
        results = detector.get_kpts(frame)
        
        # 绘制关键点
        annotated_image = detector.draw_kpts(frame, results)
        
        # 显示结果
        cv2.imshow('MediaPipe Holistic', annotated_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main_img_dir():
    # 创建图像目录
    in_img_dir = 'WLASL_images'
    out_img_dir = 'output'
    show_img = True
    os.makedirs(out_img_dir, exist_ok=True)
    detector = Detect2DKptsMediaPipe()

    img_names = os.listdir(in_img_dir)
    for i, img_name in enumerate(img_names):
        print(f'Processing image {i+1}/{len(img_names)}, {img_name}')
        img_path = os.path.join(in_img_dir, img_name)
        img = cv2.imread(img_path)
        
        # 获取关键点
        results = detector.get_kpts(img)
        
        # 绘制关键点
        annotated_image = detector.draw_kpts(img, results)
        
        out_img_name = img_name.split('.')[0] + '_2d.jpg'
        out_img_filepath = os.path.join(out_img_dir, out_img_name)
        cv2.imwrite(out_img_filepath, annotated_image)
        if i > 2:
            break

        # 显示结果
    #     cv2.imshow('MediaPipe Holistic', annotated_image)
    #     cv2.waitKey(0)

    # cv2.destroyAllWindows()


class Compare2D3DKpts():
    def __init__(self, img_dir = None, save_img_dir = None, args = None):
        self.img_dir = img_dir
        self.save_img_dir = save_img_dir
        os.makedirs(save_img_dir, exist_ok=True)
        self.detector_3d = ComputeBodyVerticesKpts(img_dir, save_img_dir)

    def compare(self, debug = False, compose_imgs = False):
        self.detector_3d.forward(debug, compare_2d = True, compose_imgs = compose_imgs)
        print(f'please check the results in {self.save_img_dir}')



# 使用示例
if __name__ == "__main__":
    pass
    # main_img_dir()
    debug = True
    debug = False
    save_img_dir = 'output2'
    # save_img_dir = 'temp_compare_2d_3d'
    comparor = Compare2D3DKpts(img_dir = '../WLASL_images', save_img_dir = save_img_dir)
    comparor.compare(debug = debug, compose_imgs = True)

    # detector_2d = Detect2DKptsMediaPipe()
    # img_path = 'output/train_00414_013_mesh.jpg'
    # detector_2d.run_on_img(img_path, './')