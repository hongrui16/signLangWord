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

class Detect2DKptsMediaPipe():
    def __init__(self):
        # 初始化 MediaPipe Holistic 模型
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False,
                                                  min_detection_confidence=0.5,
                                                  min_tracking_confidence=0.5)
        # 初始化绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.hand_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(0, 0, 255))
        self.pose_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(0, 0, 255))


    def get_kpts(self, image):
        # 处理图像，检测关键点
        results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

    def draw_kpts(self, image, results):
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.hand_drawing_spec)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.hand_drawing_spec)

        # Draw pose landmarks selectively, avoiding the face area
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if 15 <= i <= 22:
                    continue
                # if i < 11 or i > 32:  # Skip face landmarks
                # if i < 11:# or i > 32:  # Skip face landmarks
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 1, self.pose_drawing_spec.color, self.pose_drawing_spec.thickness)

        # # Draw specific facial landmarks (eyes, nose, and mouth corners)
        # if results.face_landmarks:
        #     indices = [33, 263, 61, 291, 4]  # Indices for eyes, mouth corners, and nose tip
        #     for idx in indices:
        #         part = results.face_landmarks.landmark[idx]
        #         x = int(part.x * image.shape[1])
        #         y = int(part.y * image.shape[0])
        #         cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

        return image
    
    def run_on_dir(self, in_img_dir, out_img_dir, debug = False):
        os.makedirs(out_img_dir, exist_ok=True)
        img_names = os.listdir(in_img_dir)
        for i, img_name in enumerate(img_names):
            print(f'Processing image {i+1}/{len(img_names)}, {img_name}')
            img_path = os.path.join(in_img_dir, img_name)
            img = cv2.imread(img_path)
            
            # 获取关键点
            results = self.get_kpts(img)
            
            # 绘制关键点
            annotated_image = self.draw_kpts(img, results)
            
            out_img_name = img_name.split('.')[0] + '_2d.jpg'
            out_img_filepath = os.path.join(out_img_dir, out_img_name)
            cv2.imwrite(out_img_filepath, annotated_image)
            
            if debug:
                break

        # 显示结果
        # cv2.imshow('MediaPipe Holistic', annotated_image)
        # cv2.waitKey(0)


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

        self.detector_2d = Detect2DKptsMediaPipe()
        self.detector_3d = ComputeBodyVerticesKpts(img_dir, save_img_dir)

    def compare(self, debug = False):
        self.detector_3d(debug)
        self.detector_2d.run_on_dir(self.img_dir, self.save_img_dir, debug)
        print(f'please check the results in {self.save_img_dir}')



# 使用示例
if __name__ == "__main__":
    pass
    # main_img_dir()
    comparor = Compare2D3DKpts(img_dir = 'WLASL_images', save_img_dir = 'output')
    comparor.compare(debug = True)