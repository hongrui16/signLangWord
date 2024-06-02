import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import torch.nn as nn
import platform  # Import platform module to detect the operating system


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pixie_zoo.pixie import PIXIE
from pixie_zoo.visualizer import Visualizer
from pixie_zoo.datasets.body_datasets import TestData
from pixie_zoo.utils import util
from pixie_zoo.utils.config import cfg as pixie_cfg
from pixie_zoo.utils.tensor_cropper import transform_points

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.detect_2d_kpts import Detect2DKptsMediaPipe



def rotate_mesh(mesh, rotation_degrees, axis):
    """
    对给定的网格应用旋转变换。
    :param mesh: Open3D的TriangleMesh对象。
    :param rotation_degrees: 旋转角度（度）。
    :param axis: 旋转轴，格式为[x, y, z]。
    :return: 无返回值，直接修改输入的mesh。
    """
    R = mesh.get_rotation_matrix_from_xyz(np.radians(rotation_degrees) * np.array(axis))
    mesh.rotate(R, center=(0, 0, 0))

class VisMeshPoints():
    def __init__(self, height=800, width=800, face_filepath = None, visible = False):
        self.vis = o3d.visualization.Visualizer()
        #give the code, if on windows, set visible to True
        if platform.system() == 'Windows':
            visible = visible
            print("Windows")
        elif platform.system() == 'Linux':
            visible = False
            print("Linux")

        self.vis.create_window(width=width, height=height, visible = visible)
        render_option = self.vis.get_render_option()
        if render_option is not None:
            render_option.light_on = True
            print("Render option is available.")
        else:
            print("Render option is not available.")


        self.vis.get_render_option().light_on = True
        if face_filepath is None:
            faces_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'SMPLX_NEUTRAL_2020.npz')

            # faces_filepath = '../data/SMPLX_NEUTRAL_2020.npz'
        all_data = np.load(faces_filepath, allow_pickle=True)
        self.faces = all_data['f']
        self.pcd = o3d.geometry.PointCloud()
        


    def vis_mesh(self, vertices, output_dir = None, name = None):
        mesh = o3d.geometry.TriangleMesh()

        # 设置网格的顶点
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # 设置网格的三角形面
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)

        # 旋转网格以确保正确朝向
        rotate_mesh(mesh, 180, [1, 0, 0])  # 绕X轴旋转180度

        # 计算顶点的法线
        mesh.compute_vertex_normals()

        # 重置视图
        self.vis.clear_geometries()
        self.vis.add_geometry(mesh)


        mesh_center = mesh.get_center()  # 获取网格中心
        # 调整相机焦点，将 Y 坐标稍微增加
        new_lookat = [mesh_center[0], mesh_center[1] + 0.15, mesh_center[2]]  # 根据需要调整这里的 0.1

        # # 设置相机位置
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, 1])   # 默认相机朝向
        ctr.set_lookat(new_lookat)  # 将相机焦点设置在网格的中心
        # ctr.set_lookat([0, 0, 16])  # 相机默认看向原点
        ctr.set_up([0, 1, 0])     # 保持默认的上方向
        ctr.set_zoom(0.25)  # 调整缩放比例，确保模型在视野中适当大小


        # 渲染图像
        self.vis.update_geometry(mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

        # 从 Open3D 中捕获渲染的图像
        mesh_vis = self.vis.capture_screen_float_buffer(False)
        
        # 将 Open3D 的输出转换为可用的 NumPy 图像格式
        mesh_vis = (np.asarray(mesh_vis) * 255).astype(np.uint8)
        mesh_vis = cv2.cvtColor(mesh_vis, cv2.COLOR_RGB2BGR)

        if not output_dir is None and not name is None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存当前视角下的图像
            output_filepath = os.path.join(output_dir, f'{name}_mesh.jpg')
            cv2.imwrite(output_filepath, mesh_vis)
            return output_filepath
        else:
            return None

    def vis_points(self, points, i, output_dir = None, name = None):
        self.pcd.points = o3d.utility.Vector3dVector(points)
        
        # 如果是第一次迭代，需要添加点云到可视化窗口
        if i == 0:
            self.vis.add_geometry(self.pcd)
        
        # 更新点云
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        if not output_dir is None and not name is None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存当前视角下的图像
            output_filepath = os.path.join(output_dir, f'{name}_points.jpg')
            self.vis.capture_screen_image(output_filepath)

    def destroy(self):
        self.vis.destroy_window()


class ComputeBodyVerticesKpts():
    def __init__(self, input_img_dir, save_img_dir = None, iscrop = True, args = None):
        self.input_img_dir = input_img_dir
        self.save_img_dir = save_img_dir
        self.iscrop = iscrop
        self.args = args

        ###        # 创建可视化窗口
        if platform.system() == 'Windows':
            self.mesh_point_visualizer = VisMeshPoints()
        else:
            print("Linux, need to be verified!")
            pass
        

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
        else:
            self.device = torch.device('cpu')

        if save_img_dir is not None:
            os.makedirs(save_img_dir, exist_ok=True)

        # load test images 
        self.testdata = TestData(input_img_dir, iscrop=iscrop, body_detector='rcnn')

        #-- run PIXIE
        pixie_cfg.model.use_tex = True
        self.pixie = PIXIE(config = pixie_cfg, device=self.device)

        self.detector_2d = Detect2DKptsMediaPipe()
        # point_cloud = o3d.geometry.PointCloud()
        
        # print('faces.shape', faces.shape)  
        # return

        # img_path = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\signLangWord\train_00414_013_mesh.jpg'
        # self.detector_2d.run_on_img(img_path, './', '222')



    def forward(self, debug = False, compare_2d = False, compose_imgs = False):
        for i, batch in enumerate(tqdm(self.testdata, dynamic_ncols=True)):
            util.move_dict_to_device(batch, self.device)
            batch['image'] = batch['image'].unsqueeze(0)
            batch['image_hd'] = batch['image_hd'].unsqueeze(0)
            name = batch['name']
            name = os.path.basename(name)
            name = name.split('.')[0]
            img_filepath = batch['imagepath']
            # print(name)
            # frame_id = int(name.split('frame')[-1])
            # name = f'{frame_id:05}'

            data = {
                'body': batch
            }

            param_dict = self.pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
            
            # param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=True)
            # only use body params to get smplx output. TODO: we can also show the results of cropped head/hands
            codedict = param_dict['body']

            opdict = self.pixie.decode(codedict, param_type='body')
            '''
            prediction = {
                'vertices': verts,
                'transformed_vertices': trans_verts,
                'face_kpt': projected_landmarks,
                'smplx_kpt': projected_joints,
                'smplx_kpt3d': smplx_kpt3d,
                'joints': joints,
                'cam': cam,
                }
            '''
            points = opdict['joints'].cpu().numpy().squeeze()            
            vertices = opdict['vertices'].cpu().numpy().squeeze()
            # print('points.shape', points.shape) #  (145, 3)
            # print('vertices.shape', vertices.shape) #(10475, 3)

            mesh_filepath = self.mesh_point_visualizer.vis_mesh(vertices, self.save_img_dir, name)
            # self.mesh_point_visualizer.vis_points(points, i, self.save_img_dir, name)
            if compare_2d:
                pass
                ori_img = cv2.imread(img_filepath)
                results = self.detector_2d.get_kpts(ori_img)
                annotated_image = self.detector_2d.draw_kpts(ori_img.copy(), results)
                output_2d_filepath = os.path.join(self.save_img_dir, f'{name}_2d.jpg')
                cv2.imwrite(output_2d_filepath, annotated_image)

                if compose_imgs:
                    target_size = 600
                    img_2d = cv2.imread(output_2d_filepath)
                    img_3d = cv2.imread(mesh_filepath)

                    ori_img = cv2.resize(ori_img, (target_size, target_size))
                    img_2d = cv2.resize(img_2d, (target_size, target_size))
                    img_3d = cv2.resize(img_3d, (target_size, target_size))
                    img_compose = np.hstack((ori_img, img_2d, img_3d))
                    composed_filepath = os.path.join(self.save_img_dir, f'{name}_compose.jpg')
                    cv2.imwrite(composed_filepath, img_compose)
                    os.remove(mesh_filepath)
                    os.remove(output_2d_filepath)
                # bbox, bbox_idx_dict = self.detector_2d.get_bbox(ori_img, results)
                
                # height, width = ori_img.shape[:2]
                # certer_x = bbox[0] + bbox[2] / 2
                # certer_y = bbox[1] + bbox[3] / 2
                # right = width - certer_x
                # bottom = height - certer_y
                # left = certer_x
                # top = certer_y
                # bbox_h = bbox[3] - bbox[1]
                # bbox_w = bbox[2] - bbox[0]

                # mesh_image = cv2.imread(mesh_filepath)
                # # cv2.imshow('mesh_image', mesh_image)
                # # cv2.waitKey(0)
                # # mesh_height, mesh_width = mesh_image.shape[:2]
                # mesh_results = self.detector_2d.get_kpts(mesh_image)# results = self.get_kpts(img)
                # draw_img = self.detector_2d.draw_kpts(mesh_image, mesh_results, True)
                # cv2.imshow('draw_img', draw_img)
                # cv2.waitKey(0)
                # cv2.imwrite(os.path.join('./', f'{name}_mesh_2d.jpg'), draw_img)
                # mesh_bbox = self.detector_2d.look_up_landmark_bbox(mesh_results, bbox_idx_dict)

                # mesh_certer_x = mesh_bbox[0] + mesh_bbox[2] / 2
                # mesh_certer_y = mesh_bbox[1] + mesh_bbox[3] / 2
                # # mesh_image = cv2.imread(mesh_filepath)
                # mesh_bbox_h = mesh_bbox[3] - mesh_bbox[1]
                # mesh_bbox_w = mesh_bbox[2] - mesh_bbox[0]
                # scale_x = mesh_bbox_w / bbox_w
                # scale_y = mesh_bbox_h / bbox_h
                # mesh_start_x = int(mesh_certer_x - left * scale_x)
                # mesh_start_y = int(mesh_certer_y - top * scale_y)
                # mesh_end_x = int(mesh_certer_x + right * scale_x)
                # mesh_end_y = int(mesh_certer_y + bottom * scale_y)
                # crop_mesh_image = mesh_image[mesh_start_y:mesh_end_y, mesh_start_x:mesh_end_x]
                # print('crop_mesh_image.shape', crop_mesh_image.shape)
                # cv2.imshow('crop_mesh_image', crop_mesh_image)
                # output_filepath = os.path.join(self.save_img_dir, f'{name}_mesh_crop.jpg')
                # cv2.imwrite(output_filepath, crop_mesh_image)
                


            # 结束可视化
            # image_hd = batch['image_hd'].squeeze().cpu().numpy().transpose(1, 2, 0)
            # image_hd = (image_hd * 255).astype(np.uint8)

            # projected_landmarks = opdict['face_kpt'].cpu().numpy().squeeze()
            # projected_joints = opdict['smplx_kpt'].cpu().numpy().squeeze()
            # print('projected_landmarks.shape', projected_landmarks.shape, projected_landmarks) # (468, 2)
            # print('projected_joints.shape', projected_joints.shape, projected_joints) # (468, 2)

            # img_projected_joints = image_hd.copy()[:,:,::-1]
            # img_projected_landmarks = image_hd.copy()[:,:,::-1]
            # # draw 2D projected_landmarks
            # for j in range(projected_landmarks.shape[0]):
            #     x = int(projected_landmarks[j, 0])
            #     y = int(projected_landmarks[j, 1])
            #     cv2.circle(img_projected_landmarks, (x, y), 2, (0, 255, 0), -1)                
            # output_filepath = os.path.join(self.save_img_dir, f'{name}_landmarks.jpg')
            # cv2.imwrite(output_filepath, img_projected_landmarks)

            # # draw 2D projected_joints
            # for j in range(projected_joints.shape[0]):
            #     x = int(projected_joints[j, 0])
            #     y = int(projected_joints[j, 1])
            #     cv2.circle(img_projected_joints, (x, y), 2, (0, 0, 255), -1)
            # output_filepath = os.path.join(self.save_img_dir, f'{name}_joints.jpg')
            # cv2.imwrite(output_filepath, img_projected_landmarks)




            if debug:
                break

        self.mesh_point_visualizer.destroy()
        print(f'please check the results in {self.save_img_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')
    parser.add_argument('--input_img_dir', type=str, default='../TestSamples/body', help='input image directory')
    parser.add_argument('--save_img_dir', type=str, default='output_images', help='output image directory')
    parser.add_argument('--iscrop', type=bool, default=True, help='whether crop the image')
    args = parser.parse_args()

    compute_body_vertices_kpts = ComputeBodyVerticesKpts(args.input_img_dir, args.save_img_dir, args.iscrop, args)
    compute_body_vertices_kpts.forward()