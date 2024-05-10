import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append('../')
from pixie_zoo.pixie import PIXIE
# from pixielib.pixie_parallel import PIXIE
from pixie_zoo.visualizer import Visualizer
from pixie_zoo.datasets.body_datasets import TestData
from pixie_zoo.utils import util
from pixie_zoo.utils.config import cfg as pixie_cfg
from pixie_zoo.utils.tensor_cropper import transform_points

class vis_mesh_points():
    def __init__(self, height=1000, width=1000, face_filepath = None):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height)
        self.vis.get_render_option().light_on = True
        if face_filepath is None:
            faces_filepath = 'data/SMPLX_NEUTRAL_2020.npz'
        all_data = np.load(faces_filepath, allow_pickle=True)
        self.faces = all_data['f']
        self.pcd = o3d.geometry.PointCloud()
        


    def vis_mesh(self, vertices, output_dir = None, name = None):
        mesh = o3d.geometry.TriangleMesh()

        # 设置网格的顶点
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # 设置网格的三角形面
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)

        # 计算顶点的法线
        mesh.compute_vertex_normals()

        # 重置视图
        self.vis.clear_geometries()
        self.vis.add_geometry(mesh)

        # 光照和相机视角设置
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, -0.65, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.6)

        transformation = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, np.pi))
        mesh.rotate(transformation, center=mesh.get_center())

        self.vis.update_geometry(mesh)
        self.vis.poll_events()
        self.vis.update_renderer()


        if not output_dir is None and not name is None:
            os.makedirs(output_dir, exist_ok=True)
            # 保存当前视角下的图像
            output_filepath = os.path.join(output_dir, f'{name}_mesh.jpg')
            self.vis.capture_screen_image(output_filepath)

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

def main(args):
    args.inputpath = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\smile_data\color_openpose\images'
    args.inputpath = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\expose\samples'
    args.rasterizer_type = 'pytorch3d'
    args.lightTex = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    # check env
    if not torch.cuda.is_available():
        print('CUDA is not available! use CPU instead')
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # load test images 
    testdata = TestData(args.inputpath, iscrop=args.iscrop, body_detector='rcnn')

    #-- run PIXIE
    pixie_cfg.model.use_tex = args.useTex
    pixie = PIXIE(config = pixie_cfg, device=device)

    skip_pytorch3d = True
    if skip_pytorch3d:
        visualizer = Visualizer(render_size=args.render_size, config = pixie_cfg, device=device, rasterizer_type=args.rasterizer_type)
    else:
        visualizer = None
    if args.deca_path:
        # if given deca code path, run deca to get face details, here init deca model
        sys.path.insert(0, args.deca_path)
        from decalib.deca import DECA
        deca = DECA(device=device)
        use_deca = True
    else:
        use_deca = False
        
    # point_cloud = o3d.geometry.PointCloud()
    
    # print('faces.shape', faces.shape)  
    # return
    # 创建可视化窗口
    mesh_point_visualizer = vis_mesh_points()


    for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
        util.move_dict_to_device(batch, device)
        batch['image'] = batch['image'].unsqueeze(0)
        batch['image_hd'] = batch['image_hd'].unsqueeze(0)
        name = batch['name']
        name = os.path.basename(name)
        name = name.split('.')[0]
        # print(name)
        # frame_id = int(name.split('frame')[-1])
        # name = f'{frame_id:05}'

        data = {
            'body': batch
        }
        try:
            param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
            
            # param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=True)
            # only use body params to get smplx output. TODO: we can also show the results of cropped head/hands
            moderator_weight = param_dict['moderator_weight']
            codedict = param_dict['body']

            opdict = pixie.decode(codedict, param_type='body', skip_pytorch3d = skip_pytorch3d)
            '''
            opdict = {
                        'vertices': verts,
                        'transformed_vertices': trans_verts,
                        'face_kpt': predicted_landmarks,
                        'smplx_kpt': predicted_joints,
                        'smplx_kpt3d': smplx_kpt3d,
                        'joints': joints,
                        'cam': cam,
                        }
            '''
            points = opdict['joints'].cpu().numpy().squeeze()            
            vertices = opdict['vertices'].cpu().numpy().squeeze()
            # print('points.shape', points.shape) #  (145, 3)
            # print('vertices.shape', vertices.shape) #(10475, 3)

            # mesh_point_visualizer.vis_mesh(vertices, output_dir, name)
            mesh_point_visualizer.vis_points(points, i, output_dir, name)

            # 结束可视化
            
        except Exception as e:
            continue



        if skip_pytorch3d:
            continue

        opdict['albedo'] = visualizer.tex_flame2smplx(opdict['albedo'])
        if args.saveObj or args.saveParam or args.savePred or args.saveImages or args.deca_path is not None:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        # run deca if deca is available and moderator thinks information from face crops is reliable
        if args.deca_path is not None and param_dict['moderator_weight']['head'][0,1].item()>0.6:
            cropped_face_savepath = os.path.join(savefolder, name, f'{name}_facecrop.jpg')
            cv2.imwrite(cropped_face_savepath, util.tensor2image(data['body']['head_image'][0]))
            _, deca_opdict, _ = deca.run(cropped_face_savepath)
            flame_displacement_map = deca_opdict['displacement_map']
            opdict['displacement_map'] = visualizer.tex_flame2smplx(flame_displacement_map)
        if args.lightTex:
            visualizer.light_albedo(opdict)
        if args.extractTex:
            visualizer.extract_texture(opdict, data['body']['image_hd'])
        if args.reproject_mesh and args.rasterizer_type=='standard':
            ## whether to reproject mesh to original image space
            tform = batch['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2)
            original_image = batch['original_image'][None, ...]
            visualizer.recover_position(opdict, batch, tform, original_image)
        if args.saveVis:
            if args.showWeight is False:
                moderator_weight = None 
            visdict = visualizer.render_results(opdict, data['body']['image_hd'], overlay=True, moderator_weight=moderator_weight, use_deca=use_deca)
            # show cropped parts 
            if args.showParts:
                visdict['head'] = data['body']['head_image']
                visdict['left_hand'] = data['body']['left_hand_image'] # should be flipped
                visdict['right_hand'] = data['body']['right_hand_image']
            cv2.imwrite(os.path.join(savefolder, f'{name}_vis.jpg'), visualizer.visualize_grid(visdict, size=args.render_size))
            # print(os.path.join(savefolder, f'{name}_vis.jpg'))
            # import ipdb; ipdb.set_trace()
            # exit()
        if args.saveGif:
            visualizer.rotate_results(opdict, visdict=visdict, savepath=os.path.join(savefolder, f'{name}_vis.gif'))
        if args.saveObj:
            visualizer.save_obj(os.path.join(savefolder, name, f'{name}.obj'), opdict)
        if args.saveParam:
            codedict['bbox'] = batch['bbox']
            util.save_pkl(os.path.join(savefolder, name, f'{name}_param.pkl'), codedict)
            np.savetxt(os.path.join(savefolder, name, f'{name}_bbox.txt'), batch['bbox'].squeeze())
        if args.savePred:
            util.save_pkl(os.path.join(savefolder, name, f'{name}_prediction.pkl'), opdict) 
        if args.saveImages:
            for vis_name in visdict.keys():
                cv2.imwrite(os.path.join(savefolder, name, f'{name}_{vis_name}.jpg'), util.tensor2image(visdict[vis_name][0]))
    mesh_point_visualizer.destroy()
    print(f'-- please check the results in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')

    parser.add_argument('-i', '--inputpath', default='TestSamples/body', type=str,
                        help='path to the test data, can be image folder, image path, image path list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/body/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    # rendering option
    parser.add_argument('--render_size', default=1024, type=int,
                        help='image size of renderings' )
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--reproject_mesh', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to reproject the mesh and render it in original image space, \
                            currently only available if rasterizer_type is standard, will add supports for pytorch3d \
                            after pytorch **stable version** supports non-squared images. \
                            default is False, means using the cropped image and its corresponding results')
    # texture options 
    parser.add_argument('--deca_path', default=None, type=str,
                        help='absolute path of DECA folder, if exists, will return facial details by running DECA. \
                        please refer to https://github.com/YadiraF/DECA' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--lightTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to return lit albedo: that add estimated SH lighting to albedo')
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image, only do this when the face is near frontal and very clean!')
    # save
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--showParts', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to show head/hands crops in visualization' )
    parser.add_argument('--showWeight', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to visualize the moderator weight on colored shape' )
    parser.add_argument('--saveGif', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to visualize other views of the output, save as gif' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveParam', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save parameters as pkl file' )
    parser.add_argument('--savePred', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save smplx prediction as pkl file' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())
