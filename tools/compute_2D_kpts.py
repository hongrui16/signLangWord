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
import csv

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

def save_keypoints(results, image, out_file):
    height, width, _ = image.shape

    with open(out_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['part', 'x', 'y', 'visibility'])
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                writer.writerow(['pose_' + str(idx), x_px, y_px, landmark.visibility])
        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                writer.writerow(['left_hand_' + str(idx), x_px, y_px, landmark.visibility])
        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                writer.writerow(['right_hand_' + str(idx), x_px, y_px, landmark.visibility])





def main_img_dir():
    # 创建图像目录
    in_img_dir = '../WLASL_images'
    out_img_dir = 'output_csv'
    os.makedirs(out_img_dir, exist_ok=True)
    detector = Detect2DKptsMediaPipe()

    img_names = os.listdir(in_img_dir)
    for i, img_name in enumerate(img_names):
        print(f'Processing image {i+1}/{len(img_names)}, {img_name}')
        img_path = os.path.join(in_img_dir, img_name)
        img = cv2.imread(img_path)
        results = detector.get_kpts(img)

        # 保存关键点到CSV文件
        output_filepath = os.path.join(out_img_dir, os.path.splitext(img_name)[0] + '.csv')
        save_keypoints(results, img, output_filepath)

        if i > 2:
            break




# 使用示例
if __name__ == "__main__":
    pass
    main_img_dir()
