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



class Detect2DKptsMediaPipe():
    def __init__(self):
        # 初始化 MediaPipe Holistic 模型
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False,
                                                  min_detection_confidence=0.08,
                                                  min_tracking_confidence=0.1)
        # 初始化绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.hand_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(0, 0, 255))
        self.pose_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(0, 0, 255))


    def get_kpts(self, image):
        # 处理图像，检测关键点
        results  = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        return results

    def get_bbox(self, image, results):
        h, w, _ = image.shape
        x_min, x_max = w, 0
        y_min, y_max = h, 0
        bbox_idx_dict = {'xmin': -1, 'ymin': -1, 'xmax': -1, 'ymax': -1}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            ## must be visible
            if landmark.visibility < 0.5:
                continue
            x, y = int(landmark.x * w), int(landmark.y * h)
            if x < x_min:
                x_min = x
                bbox_idx_dict['xmin'] = idx
            if x > x_max:
                x_max = x
                bbox_idx_dict['xmax'] = idx
            if y < y_min:
                y_min = y
                bbox_idx_dict['ymin'] = idx
            if y > y_max:
                y_max = y
                bbox_idx_dict['ymax'] = idx
        return [x_min, y_min, x_max, y_max], bbox_idx_dict
    
    def look_up_landmark_bbox(self, results, bbox_idx_dict):
        ## convert landmarks to a list first
        landmark_x_list = []
        landmark_y_list = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_x_list.append(landmark.x)
            landmark_y_list.append(landmark.y)
        ## look up bbox landmarks
        bbox_landmarks = []
        bbox_landmarks.append(landmark_x_list[bbox_idx_dict['xmin']])
        bbox_landmarks.append(landmark_y_list[bbox_idx_dict['ymin']])
        bbox_landmarks.append(landmark_x_list[bbox_idx_dict['xmax']])
        bbox_landmarks.append(landmark_y_list[bbox_idx_dict['ymax']])
        return bbox_landmarks
    
    def draw_kpts(self, image, results, show_img = False):
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
        if show_img:
            cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(5) & 0xFF == 27:
                return image

        return image

    def run_on_img(self, img_path, out_img_dir = None, prefix = None):
        if not os.path.exists(img_path):
            print(f'Image {img_path} does not exist')
            return 
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        # 获取关键点
        results = self.get_kpts(img)
        if out_img_dir is None:
            print('No output directory specified')
            return 
                
        os.makedirs(out_img_dir, exist_ok=True)
        # 绘制关键点
        annotated_image = self.draw_kpts(img, results)
        
        if not prefix is None:
            out_img_name = img_name.split('.')[0] + f'_{prefix}_2d.jpg'
        else:
            out_img_name = img_name.split('.')[0] + '_2d.jpg'
        out_img_filepath = os.path.join(out_img_dir, out_img_name)
        cv2.imwrite(out_img_filepath, annotated_image)
        print(f'Saved to {out_img_filepath}')
        
    def run_on_dir(self, in_img_dir, out_img_dir, debug = False):
        os.makedirs(out_img_dir, exist_ok=True)
        img_names = os.listdir(in_img_dir)
        for i, img_name in enumerate(img_names):
            print(f'Processing image {i+1}/{len(img_names)}, {img_name}')
            img_path = os.path.join(in_img_dir, img_name)
            
            self.run_on_img(img_path, out_img_dir)
            
            if debug:
                break
        print(f'Processed {len(img_names)} images, saved to {out_img_dir}')


# 使用示例
if __name__ == "__main__":
    pass
    # main_img_dir()
    # comparor = Compare2D3DKpts(img_dir = '../WLASL_images', save_img_dir = 'output')
    # comparor.compare(debug = True)

    detector_2d = Detect2DKptsMediaPipe()
    img_path = '../tools/output/train_00414_013_mesh.jpg'
    detector_2d.run_on_img(img_path, './')