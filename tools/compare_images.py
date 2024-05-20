import os, sys
import cv2
import numpy as np


def compose_images(in_dir1, in_dir2, out_dir):
    # 创建输出目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 获取目录中的文件名
    img_names1 = os.listdir(in_dir1)
    img_names2 = os.listdir(in_dir2)

    # 读取图像
    for img_name1, img_name2 in zip(img_names1, img_names2):
        img1 = cv2.imread(os.path.join(in_dir1, img_name1))
        img2 = cv2.imread(os.path.join(in_dir2, img_name2))
        
        # 检查图像是否读取成功
        if img1 is None or img2 is None:
            print(f'Error: image {img_name1} or {img_name2} not found')
            continue
        h1, w1, _ = img1.shape
        img1_right = img1[:, w1//2:, :]
        new_h, new_w, _ = img1_right.shape
        img2 = cv2.resize(img2, (3*new_w, new_h))
        
        #draw text on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,30)
        fontScale = 1
        fontColor = (0,255,0)
        lineType = 2
        img1_right = cv2.putText(img1_right, 'SMPLer-X', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
        img2_right = img2[:, 2*new_w:, :]
        img2_right = cv2.putText(img2_right, 'PIXIE', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        new_img = np.hstack((img2[:, :2*new_w], img2_right, img1_right))

        
        # 保存结果
        out_name = img_name1.split('.')[0] + '.jpg'
        cv2.imwrite(os.path.join(out_dir, out_name), new_img)

    print(f'Images composed successfully in {out_dir}')

def show_images_with_html(input_dir):

    img_names = os.listdir(input_dir)
    # display an image in a clumn
    html = '<html><head></head><body>'
    for img_name in img_names:
        img_path = os.path.join(input_dir, img_name)
        html += f'<img src="{img_path}" width="1000">'
    html += '</body></html>'
    with open('output.html', 'w') as f:
        f.write(html)
    print('HTML file created successfully')

if __name__ == "__main__":
    in_dir1 = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\SMPLer-X\output\test_20240514_033038\result\vis'
    in_dir2 = 'output2'
    out_dir = 'output_composed'
    # compose_images(in_dir1, in_dir2, out_dir)
    show_images_with_html(out_dir)