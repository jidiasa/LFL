import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import your existing modules:
# - Scene and GaussianModel come from scene.py
# - GSParams and CameraParams from arguments.py (or wherever they are defined)
# - render function from gaussian_renderer.py
# - colorize from utils.depth
# - l1_loss, ssim from utils.loss
# etc.
import sys
sys.path.append("/home/ubuntu/Desktop/WonderJourney/models/GSRefiner")
from scene import Scene, GaussianModel
from arguments import GSParams, CameraParams
from gaussian_renderer import render
from models.GSRefiner.utils.loss import l1_loss, ssim
from models.GSRefiner.utils.depth import colorize

import os
import glob
import json
import time
import datetime
import warnings
import shutil
from random import randint
from argparse import ArgumentParser

warnings.filterwarnings(action='ignore')

import pickle
import imageio
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter

import torch
import torch.nn.functional as F
import gradio as gr
from diffusers import (
    StableDiffusionInpaintPipeline, StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline)

from arguments import GSParams, CameraParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.dataset_readers import loadCameraPreset
from models.GSRefiner.utils.loss import l1_loss, ssim
from models.GSRefiner.utils.camera import load_json
from models.GSRefiner.utils.depth import colorize
from models.GSRefiner.utils.lama import LaMa
from models.GSRefiner.utils.trajectory import get_camerapaths, get_pcdGenPoses
from scipy.spatial.transform import Rotation as R, Slerp

get_kernel = lambda p: torch.ones(1, 1, p * 2 + 1, p * 2 + 1).to('cuda')
t2np = lambda x: (x[0].permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
np2t = lambda x: (torch.as_tensor(x).to(torch.float32).permute(2, 0, 1) / 255.0)[None, ...].to('cuda')
pad_mask = lambda x, padamount=1: t2np(
F.conv2d(np2t(x[..., None]), get_kernel(padamount), padding=padamount))[..., 0].astype(bool)

def weight_schedule(iteration, start_iter, end_iter, start_weight, end_weight):
    if iteration < start_iter:
        return start_weight
    elif iteration > end_iter:
        return end_weight
    else:
        return start_weight + (end_weight - start_weight) * (iteration - start_iter) / (end_iter - start_iter)

import shutil

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    # grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    # grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=0.0).squeeze()
    return grad_img

def interpolate_w2c_fixed_length(w2c_matrices, start_index):
    """
    从 start_index 到列表末尾，插值生成与原始列表长度一致的 w2c 序列。
    前 start_index 帧保持不变，后面部分为插值结果。

    参数:
        w2c_matrices: List[np.ndarray]
            原始的 w2c 4x4 矩阵列表
        start_index: int
            从第几个帧开始插值（如 start_index=5 表示第6帧作为插值起点）

    返回:
        List[np.ndarray]
            长度与原始相同的新 w2c 列表，前段为原始，后段为插值结果
    """
    total_len = len(w2c_matrices)
    assert 0 <= start_index < total_len - 1, "start_index 越界或为最后一帧"

    # 起止 w2c
    w2c_start = w2c_matrices[start_index]
    w2c_end = w2c_matrices[-1]
    interp_len = total_len - start_index

    # 分解旋转和平移
    R0 = w2c_start[:3, :3]
    R1 = w2c_end[:3, :3]
    t0 = w2c_start[:3, 3]
    t1 = w2c_end[:3, 3]

    # 构造旋转插值器
    rotations = R.from_matrix([R0, R1])
    slerp = Slerp([0, 1], rotations)

    # 插值时间点
    times = np.linspace(0, 1, interp_len)
    interp_rots = slerp(times)
    interp_trans = np.outer(1 - times, t0) + np.outer(times, t1)

    # 构造插值矩阵
    interpolated = []
    for i in range(interp_len):
        mat = np.eye(4)
        mat[:3, :3] = interp_rots[i].as_matrix()
        mat[:3, 3] = interp_trans[i]
        interpolated.append(mat)

    return interpolated

def get_normal_diff(normal):
    _, hd, wd = normal.shape 
    bottom_point = normal[..., 2:hd,   1:wd-1]
    top_point    = normal[..., 0:hd-2, 1:wd-1]
    right_point  = normal[..., 1:hd-1, 2:wd]
    left_point   = normal[..., 1:hd-1, 0:wd-2]
    grad_img_x = 1 - (right_point * left_point).sum(dim=0,keepdim=True)
    grad_img_y = 1 - (top_point * bottom_point).sum(dim=0,keepdim=True)
    # grad_img_x = torch.where(grad_img_x > 0, 1 - grad_img_x, grad_img_x)
    # grad_img_y = torch.where(grad_img_y > 0, 1 - grad_img_y, grad_img_y)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img = torch.mean(grad_img, dim=0)
    # grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    # grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=0.0).squeeze()
    # center_point = normal[..., 1:hd-1, 1:wd-1]
    # diff_img_x = 1 - (center_point * left_point).sum(dim=0,keepdim=True)
    # diff_img_y = 1 - (center_point * bottom_point).sum(dim=0,keepdim=True)
    # diff_img = (diff_img_x + diff_img_y) / 2
    
    return grad_img 
# def render_point_cloud(point_cloud, C2W, point_colors=None, save_dir="debug/output/rendered_images", img_name="rendered.png", 
#                        img_size=(512, 512), fx=500, fy=500):
#     os.makedirs(save_dir, exist_ok=True)

#     W2C = np.linalg.inv(C2W)  

#     num_points = point_cloud.shape[0]
#     points_h = np.hstack((point_cloud, np.ones((num_points, 1))))  
#     points_camera = (W2C @ points_h.T).T[:, :3]  

#     valid_mask = points_camera[:, 2] > 0  
#     points_camera_valid = points_camera[valid_mask]

#     if point_colors is not None:
#         if isinstance(point_colors, tuple):  
#             point_colors_valid = np.tile(np.array(point_colors), (len(points_camera_valid), 1))
#         else:
#             point_colors_valid = point_colors[valid_mask]
#     else:
#         point_colors_valid = points_camera_valid[:, 2]

#     x = points_camera_valid[:, 0] / points_camera_valid[:, 2]
#     y = points_camera_valid[:, 1] / points_camera_valid[:, 2]
#     cx, cy = img_size[0] / 2, img_size[1] / 2
#     K = np.array([
#         [fx,  0, cx],
#         [ 0, fy, cy],
#         [ 0,  0,  1]
#     ])

#     pixel_coords = (K @ np.vstack((x, y, np.ones_like(x)))).T
#     pixel_x = pixel_coords[:, 0] / pixel_coords[:, 2]
#     pixel_y = pixel_coords[:, 1] / pixel_coords[:, 2]

#     plt.figure(figsize=(8, 6))
#     plt.scatter(pixel_x, pixel_y, s=1, c=point_colors_valid, cmap='jet' if point_colors is None else None)
#     plt.xlim(0, img_size[0])
#     plt.ylim(img_size[1], 0)  
#     plt.xlabel("Pixel X")
#     plt.ylabel("Pixel Y")
#     plt.title("Rendered Image from Point Cloud")
    
#     if point_colors is None:
#         plt.colorbar(label="Depth (Z)")

#     save_path = os.path.join(save_dir, img_name)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close() 

#     print(f"Rendered image saved at: {save_path}")

# class GaussSplatRefiner:

#     def __init__(
#         self,
#         gs_params=None,
#         camera_params=None,
#         white_background=False,
#         output_dir="outputs",
#     ):
#         self.opt = gs_params if gs_params is not None else GSParams()
#         self.cam = camera_params if camera_params is not None else CameraParams()

#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.gaussians = GaussianModel(self.opt.sh_degree)

#         bg_color = [1, 1, 1] if white_background else [0, 0, 0]
#         self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#         self.scene = None

#     def load_scene(self, pcd_points, pcd_colors, frames):
#         # Create a dictionary in the same style used by the LucidDreamer pipeline
#         # So that Scene(...) can read it properly
#         traindata = {
#             "camera_angle_x": self.cam.fov[0],  # or provide your own if needed
#             "W": self.cam.W,
#             "H": self.cam.H,
#             "pcd_points": pcd_points,
#             "pcd_colors": pcd_colors,
#             "frames": frames,
#         }

#         self.scene = Scene(traindata, self.gaussians, self.opt)

#     def optimize_scene(self, save_render_result=False):
#         if self.scene is None:
#             raise ValueError("Scene not loaded. Call load_scene(...) first.")

#         # Shortcut references
#         gaussians = self.gaussians
#         opt = self.opt
#         background = self.background
#         print("opacities.shape =", self.gaussians._opacity.shape)
#         print("opacities min =",self.gaussians._opacity.min().item())
#         print("opacities max =", self.gaussians._opacity.max().item())


#         # Train cameras from the scene
#         output_root = "debug/output/render"
#         os.makedirs(output_root, exist_ok=True)
#         to_pil = transforms.ToPILImage()

#         gaussians.optimizer.zero_grad(set_to_none=True)

#         my_points = np.load("debug/my_points.npy") 
#         my_colors = np.load("debug/my_colors.npy")  

#         for iteration in tqdm(range(1, opt.iterations + 1), desc="Optimizing Gaussians"):
#             # Update learning rate
#             gaussians.update_learning_rate(iteration)

#             # Possibly increase spherical harmonic degree every 1000 steps
#             if iteration % 1000 == 0:
#                 gaussians.oneupSHdegree()

#             # Random camera selection from the training set
#             viewpoint_stack = self.scene.getTrainCameras().copy()
#             viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
#             # Render from that viewpoint
#             # render_point_cloud(my_points, viewpoint_cam., my_colors)
#             render_pkg = render(viewpoint_cam, gaussians, opt, background)
#             image_rendered = render_pkg["render"]  # (C,H,W)
#             visibility_filter = render_pkg["visibility_filter"]
#             viewspace_pts = render_pkg["viewspace_points"]
#             radii = render_pkg["radii"]

#             # Ground truth
#             gt_image = viewpoint_cam.original_image.cuda()

#             if iteration % 50 == 1 and save_render_result == True:
#                 iter_dir = os.path.join(output_root, f"iteration_{iteration}")
#                 os.makedirs(iter_dir, exist_ok=True)

#                 # 转换为 PIL 并保存
#                 render_pil = to_pil(image_rendered.cpu().clamp(0, 1))
#                 gt_pil = to_pil(gt_image.cpu().clamp(0, 1))

#                 render_pil.save(os.path.join(iter_dir, "rendered.png"))
#                 gt_pil.save(os.path.join(iter_dir, "gt_image.png"))

#             # Compute loss
#             loss_l1 = l1_loss(image_rendered, gt_image)
#             loss_ssim = 1.0 - ssim(image_rendered, gt_image)
#             loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * loss_ssim

#             # Backprop
#             loss.backward()

#             with torch.no_grad():
#                 # Densification logic
#                 if iteration < opt.densify_until_iter:
#                     # Keep track of the maximum radii in image-space
#                     gaussians.max_radii2D[visibility_filter] = torch.max(
#                         gaussians.max_radii2D[visibility_filter],
#                         radii[visibility_filter],
#                     )
#                     # Collect gradient stats for possible densification
#                     gaussians.add_densification_stats(viewspace_pts, visibility_filter)

#                     # Actually densify + prune
#                     if (iteration > opt.densify_from_iter and
#                         iteration % opt.densification_interval == 0):
#                         size_threshold = (
#                             20 if iteration > opt.opacity_reset_interval else None
#                         )
#                         gaussians.densify_and_prune(
#                             opt.densify_grad_threshold,
#                             0.005,
#                             self.scene.cameras_extent,
#                             size_threshold,
#                         )

#                     # Reset opacity periodically
#                     if (iteration % opt.opacity_reset_interval == 0 or
#                         (opt.white_background and iteration == opt.densify_from_iter)):
#                         gaussians.reset_opacity()

#                 # Step the optimizer
#                 if iteration < opt.iterations:
#                     gaussians.optimizer.step()
#                     gaussians.optimizer.zero_grad(set_to_none=True)

#     def save_ply(self, filename=None):
#         directory = os.path.dirname(filename)  # Get the parent directory of the file
#         os.makedirs(directory, exist_ok=True)
#         filename = os.path.join(self.output_dir, "gsplat.ply")
#         self.gaussians.save_ply(filename)
#         print(f"Saved PLY to: {filename}")

#     def render_video(self, camera_path, video_path="output", fps=60):
#         """
#         Renders a video (and optionally a depth video) from a set of camera configurations.

#         Args:
#             camera_path (str or list): 
#                 - If str, path to a JSON file describing the camera path, or 
#                 - If list, a list of camera dictionaries (transform, intrinsics).
#             video_path (str): Path where the RGB video will be saved.
#             depth_path (str): Path where the depth video will be saved.
#             fps (int): Frames per second for the output.
#         """
#         if self.scene is None:
#             raise ValueError("Scene not loaded. Call load_scene(...) first.")

#         # If camera_path is a JSON file, let the Scene parse that
#         if isinstance(camera_path, str):
#             views = self.scene.getPresetCameras(camera_path)
#         else:
#             # Otherwise assume it's a list of camera viewpoints
#             views = camera_path

#         frames_rgb = []

#         print("Rendering frames for video...")
#         for cam_view in tqdm(views, desc="Rendering"):
#             render_pkg = render(cam_view, self.gaussians, self.opt, self.background)
#             frame_rgb = render_pkg["render"]  # (C,H,W)

#             # Convert to numpy
#             frame_rgb_np = (
#                 frame_rgb.permute(1,2,0).clamp(0,1).detach().cpu().numpy() * 255
#             ).astype(np.uint8)
#             frames_rgb.append(frame_rgb_np)

#         directory = os.path.dirname(video_path)  # Get the parent directory of the file
#         os.makedirs(directory, exist_ok=True)
#         filename = os.path.join(self.output_dir, "output.mp4")
#         print("Writing video files...")
#         imageio.mimwrite(filename, frames_rgb, fps=fps, quality=8)
#         print(f"Videos saved: {video_path}")

def save_traindata(traindata, save_path="traindata.pkl"):
    """
    保存 traindata 到指定路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 处理 `PIL.Image` 为 numpy 数组)  # 转换为 numpy 数组

    # 使用 pickle 保存
    with open(save_path, "wb") as f:
        pickle.dump(traindata, f)
    
    print(f"Traindata saved successfully to {save_path}")

def load_traindata(load_path="traindata.pkl"):
    """
    读取 traindata
    """
    with open(load_path, "rb") as f:
        traindata = pickle.load(f)

    # 处理 numpy 数组恢复为 `PIL.Image` # 重新转换为 PIL Image

    print(f"Traindata loaded successfully from {load_path}")
    return traindata

class GaussSplatRefiner:
    def __init__(self, for_gradio=True, save_dir=None):
        print("start init")
        self.opt = GSParams()
        print("start init")
        self.cam = CameraParams()
        print("start init")
        self.save_dir = save_dir
        self.for_gradio = for_gradio
        print("start init")
        self.root = 'outputs'
        self.default_model = 'SD1.5 (default)'
        # self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        print("start init")
        self.gaussians = GaussianModel(self.opt.sh_degree)

        bg_color = [1, 1, 1] if self.opt.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        print("start init")
        self.d_model = torch.hub.load('models/GSRefiner/ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
        print("start init")
        self.controlnet = None
        self.lama = None

    # def load_model(self, model_name, use_lama=True):
    #     if model_name is None:
    #         model_name = self.default_model
    #     if self.current_model == model_name:
    #         return
    #     if model_name == self.default_model:
    #         self.controlnet = None
    #         self.lama = None
    #         self.rgb_model = StableDiffusionInpaintPipeline.from_pretrained(
    #             #  'runwayml/stable-diffusion-inpainting',
    #             'stablediffusion/SD1-5',
    #             revision='fp16',
    #             torch_dtype=torch.float16,
    #             safety_checker=None,
    #         ).to('cuda')
    #     else:
    #         if self.controlnet is None:
    #             self.controlnet = ControlNetModel.from_pretrained(
    #                 'lllyasviel/control_v11p_sd15_inpaint', torch_dtype=torch.float16)
    #         if self.lama is None and use_lama:
    #             self.lama = LaMa('cuda')
    #         self.rgb_model = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    #             f'stablediffusion/{model_name}',
    #             controlnet=self.controlnet,
    #             revision='fp16',
    #             torch_dtype=torch.float16,
    #             safety_checker=None,
    #         ).to('cuda')
    #         # self.rgb_model.enable_model_cpu_offload()
    #     torch.cuda.empty_cache()
    #     self.current_model = model_name

    # def rgb(self, prompt, image, negative_prompt='', generator=None, num_inference_steps=50, mask_image=None):
    #     image_pil = Image.fromarray(np.round(image * 255.).astype(np.uint8))
    #     mask_pil = Image.fromarray(np.round((1 - mask_image) * 255.).astype(np.uint8))
    #     if self.current_model == self.default_model:
    #         return self.rgb_model(
    #             prompt=prompt,
    #             negative_prompt=negative_prompt,
    #             generator=generator,
    #             num_inference_steps=num_inference_steps,
    #             image=image_pil,
    #             mask_image=mask_pil,
    #         ).images[0]

    #     kwargs = {
    #         'negative_prompt': negative_prompt,
    #         'generator': generator,
    #         'strength': 0.9,
    #         'num_inference_steps': num_inference_steps,
    #         'height': self.cam.H,
    #         'width': self.cam.W,
    #     }

    #     image_np = np.round(np.clip(image, 0, 1) * 255.).astype(np.uint8)
    #     mask_sum = np.clip((image.prod(axis=-1) == 0) + (1 - mask_image), 0, 1)
    #     mask_padded = pad_mask(mask_sum, 3)
    #     masked = image_np * np.logical_not(mask_padded[..., None])

    #     if self.lama is not None:
    #         lama_image = Image.fromarray(self.lama(masked, mask_padded).astype(np.uint8))
    #     else:
    #         lama_image = image

    #     mask_image = Image.fromarray(mask_padded.astype(np.uint8) * 255)
    #     control_image = self.make_controlnet_inpaint_condition(lama_image, mask_image)

    #     return self.rgb_model(
    #         prompt=prompt,
    #         image=lama_image,
    #         control_image=control_image,
    #         mask_image=mask_image,
    #         **kwargs,
    #     ).images[0]

    def d(self, im):
        return self.d_model.infer_pil(im)

    def run(self, images, camerapath, seed, diff_steps, example_name=None):
        gaussians = self.create(
            images, camerapath, seed, diff_steps, example_name)
        gallery, depth = self.render_video(camerapath, example_name=example_name)
        return (gaussians, gallery, depth)
    
    def create_scene(self, traindata_path=None, example_name=None):
        outfile = os.path.join('examples', f'{example_name}.ply')
        self.traindata = load_traindata(traindata_path)
        self.scene = Scene(self.traindata, self.gaussians, self.opt)
        #self.training()
        return

    def create(self, camerapath, pts_coord_world, pts_colors, imgs = [], example_name=None, traindata_path=None, generate=True):

        outfile = os.path.join('examples', f'{example_name}.ply')
        if generate == True:
            self.traindata = self.debug_images(camerapath, pts_coord_world, pts_colors, imgs)
        else:
            self.traindata = load_traindata(traindata_path)
        self.scene = Scene(self.traindata, self.gaussians, self.opt)         
        self.training()
        # outfile = self.save_ply(outfile)
        return outfile
    
    def create_from_frames(self, frames, pc, pc_colors):
        H, W, K = self.cam.H, self.cam.W, self.cam.K
        traindata = {
            'camera_angle_x': self.cam.fov[0],
            'W': W,
            'H': H,
            'pcd_points': pc,
            'pcd_colors': pc_colors,
            'frames': frames,
        }
        self.traindata = traindata
        self.scene = Scene(self.traindata, self.gaussians, self.opt)
        self.training()
        return
    
    def save_ply(self, fpath=None):
        if fpath is None:
            dpath = os.path.join(self.root, self.timestamp)
            fpath = os.path.join(dpath, 'gsplat.ply')
            os.makedirs(dpath, exist_ok=True)
        if not os.path.exists(fpath):
            self.gaussians.save_ply(fpath)
        else:
            self.gaussians.load_ply(fpath)
        return fpath

    def render_video(self, example_name=None, fps=30, camerapath=0, progress=gr.Progress()):
        
        videopath = os.path.join(self.save_dir, f'{example_name}.mp4')
        frame_dir = Path("ablation")
        os.makedirs(frame_dir, exist_ok=True)
        
        if camerapath == 0:
            views = self.scene.getRenderCameras().copy()
        else:
            views = self.scene.getTrainCameras().copy()
        
        framelist = []
        depthlist = []
        dmin, dmax = 1e8, -1e8

        if self.for_gradio:
            iterable_render = progress.tqdm(views, desc='[4/4] Rendering a video')
        else:
            iterable_render = views

        idx = 0

        for view in iterable_render:
            results = render(view, self.gaussians, self.opt, self.background)
            frame = results['render']
            frame_np = np.round(frame.permute(1, 2, 0).detach().cpu().numpy()[::-1].clip(0, 1) * 255.).astype(np.uint8)
            # plt.imsave("frame.png", frame.permute(1,2,0).detach().cpu().numpy().clip(0,1))
            framelist.append(frame_np)
            
                # 保存当前帧为图片
            image_path = os.path.join(frame_dir, f"frame_gs_{idx:04d}.png")

            idx = idx + 1
            Image.fromarray(frame_np).save(image_path)


        progress(1, desc='[4/4] Rendering a video...')

        # depthlist = [colorize(depth, vmin=dmin, vmax=dmax) for depth in depthlist]
        depthlist = [colorize(depth) for depth in depthlist]
        imageio.mimwrite(videopath, framelist, fps=fps, quality=8)
        return videopath
    
    def training(self, progress=gr.Progress()):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        if self.for_gradio:
            iterable_gauss = progress.tqdm(range(1, self.opt.iterations + 1), desc='[3/4] Baking Gaussians')
        else:
            iterable_gauss = range(1, self.opt.iterations + 1)

        for iteration in iterable_gauss:
            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            # viewpoint_cam = viewpoint_stack[0]

            # import pdb; pdb.set_trace()
            # Render
            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

            if iteration == 2990:
                pass

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()

            if iteration < 50:
                iter_dir = os.path.join(self.save_dir, f"iteration_{iteration}")
                os.makedirs(iter_dir, exist_ok=True)
                to_pil = transforms.ToPILImage()

                # 转换为 PIL 并保存
                render_pil = to_pil(image.cpu().clamp(0, 1))
                gt_pil = to_pil(gt_image.cpu().clamp(0, 1))

                render_pil.save(os.path.join(iter_dir, "rendered.png"))
                gt_pil.save(os.path.join(iter_dir, "gt_image.png"))
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            lambda_normal = 0.2 if iteration > 700 else 0.0
            # lambda_dist = opt.lambda_dist if iteration > 3000 and iteration < 15000 else 0.0
            lambda_dist = 0.2 if iteration > 300 else 0.0
            lambda_edge = weight_schedule(iteration, 900, 1500, 0.0, 1.0) * 0.5
            lambda_smooth = weight_schedule(iteration, 900, 1500, 0.0, 1.0) * 0.5
            lambda_kappa = 0.001 if iteration > 300 else 0.0
            rend_dist = render_pkg["rend_dist"] 
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None] 
            normal_loss = lambda_normal * (normal_error).mean() if lambda_normal > 0.0 else 0.0

            dist_loss = lambda_dist * (rend_dist).mean()
            
            # if lambda_dist > 0.0:
            #     print(f"dist_loss: {dist_loss.item()}")
            grad_image = get_img_grad_weight(gt_image)
            # grad_image = erode(grad_image > 0.1)
            edge_mask = grad_image.reshape(-1) > 0.2
            grad_normal = get_normal_diff(rend_normal).reshape(-1)
            
            mask = ~grad_normal.isnan()
            edge_mask = edge_mask[mask]
            grad_normal = grad_normal[mask]

            # depth_diff = depth_diff[mask]
            edge_loss = (2 - grad_normal[edge_mask]).mean() if lambda_edge > 0.0 else 0.0
            # scale_mask = gaussians.get_scaling[visibility_filter][:,2] > 0.2
            smooth_loss = lambda_smooth * grad_normal[~edge_mask].mean() if lambda_smooth > 0.0 else 0.0
            loss = loss + normal_loss + lambda_edge * edge_loss + smooth_loss + dist_loss
            
            loss.backward()

            with torch.no_grad():
                # Densification
                if iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

    def generate_pcd(self, camerapath, pts_coord_world, pts_colors, progress=gr.Progress()):
        ## processing inputs

        H, W, K = self.cam.H, self.cam.W, self.cam.K
        edgeN = 2
        edgemask = np.ones((H-2*edgeN, W-2*edgeN))
        edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))

        render_poses = camerapath

        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
        traindata = {
            'camera_angle_x': self.cam.fov[0],
            'W': W,
            'H': H,
            'pcd_points': pts_coord_world,
            'pcd_colors': pts_colors,
            'frames': [],
        }

        # render_poses = get_pcdGenPoses(pcdgenpath)
        internel_render_poses = get_pcdGenPoses('hemisphere', {'center_depth': 2.5})

        if self.for_gradio:
            progress(0, desc='[2/4] Aligning...')
            iterable_align = progress.tqdm(range(len(render_poses)), desc='[2/4] Aligning')
        else:
            iterable_align = range(len(render_poses))

        for i in iterable_align:
            # internel_render_poses = internel_render_poses_list[i]
            for j in range(len(internel_render_poses)):
                idx = i * len(internel_render_poses) + j
                print(f'{idx+1} / {len(render_poses)*len(internel_render_poses)}')

                ### Transform world to pixel
                Rw2i = render_poses[i,:3,:3]
                Tw2i = render_poses[i,:3,3:4]
                Ri2j = internel_render_poses[j,:3,:3]
                Ti2j = internel_render_poses[j,:3,3:4]

                Rw2j = np.matmul(Ri2j, Rw2i)
                Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j

                # Transfrom cam2 to world + change sign of yz axis
                Rj2w = np.matmul(yz_reverse, Rw2j).T
                Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
                Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

                pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
                pixel_coord_camj = np.matmul(K, pts_coord_camj)

                valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]<=W-1, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]<=H-1)))[0]
                if len(valid_idxj) == 0:
                    continue
                pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
                pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
                round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)

                print("Rw2j")
                print(Rw2j)
                print("Tw2j")
                print(Tw2j)
                print("K")
                print(K)


                x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
                grid = np.stack((x,y), axis=-1).reshape(-1,2)
                imagej = interp_grid(pixel_coord_camj.transpose(1,0), pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(H,W,3)
                imagej = edgemask[...,None]*imagej + (1-edgemask[...,None])*np.pad(imagej[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')
                img = Image.fromarray(np.round(imagej*255.).astype(np.uint8))
                output_path = f"debug/output/image_at0.png"
                img.save(output_path)

                depthj = interp_grid(pixel_coord_camj.transpose(1,0), pts_depthsj.T, grid, method='linear', fill_value=0).reshape(H,W)
                depthj = edgemask*depthj + (1-edgemask)*np.pad(depthj[1:-1,1:-1], ((1,1),(1,1)), mode='edge')

                maskj = np.zeros((H,W), dtype=np.float32)
                maskj[round_coord_camj[1], round_coord_camj[0]] = 1
                maskj = maximum_filter(maskj, size=(9,9), axes=(0,1))
                imagej = maskj[...,None]*imagej + (1-maskj[...,None])*(-1)

                maskj = minimum_filter((imagej.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
                imagej = maskj[...,None]*imagej + (1-maskj[...,None])*0

                img = Image.fromarray(np.round(imagej*255.).astype(np.uint8))
                output_path = f"debug/output/image_{i}_{j}.png"
                img.save(output_path)

                print(pts_coord_world[:,:100])
                print(pts_colors[:100])

                traindata['frames'].append({
                    'image': Image.fromarray(np.round(imagej*255.).astype(np.uint8)), 
                    'transform_matrix': Pc2w.tolist(),
                })
        
        save_data = True
        traindata_temp = traindata
        if save_data:
            path = os.path.join(self.save_dir, "traindata.pkl")
            save_traindata(traindata_temp, save_path=path)
        progress(1, desc='[3/4] Baking Gaussians...')
        return traindata

    def debug_images(self, render_poses, pts_coord_world, pts_colors, imgs, progress=gr.Progress()):

        H, W, K = self.cam.H, self.cam.W, self.cam.K
        edgeN = 2
        edgemask = np.ones((H-2*edgeN, W-2*edgeN))
        edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))

        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

        traindata = {
            'camera_angle_x': self.cam.fov[0],
            'W': W,
            'H': H,
            'pcd_points': pts_coord_world,
            'pcd_colors': pts_colors,
            'frames': [],
        }
        print(len(imgs))
        print(len(render_poses))

        if self.for_gradio:
            progress(0, desc='[2/4] Rendering...')
            iterable = progress.tqdm(enumerate(render_poses), desc='[2/4] Rendering')
        else:
            iterable = enumerate(render_poses)

        for idx, pose in iterable:
            print(f'Rendering view {idx+1}/{len(render_poses)}')

            Rw2j = pose[:3, :3]
            Tw2j = pose[:3, 3:4]

            # 计算变换矩阵: 相机到世界 + YZ轴反转
            Rj2w = np.matmul(yz_reverse, Rw2j).T
            Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
            Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)
            M = np.array([[1, 0, 0],
              [0, -1, 0],
              [0,  0, -1]])  # OpenGL -> OpenCV

            # Rw2j = np.matmul(M, Rw2j)
            # Tw2j = np.matmul(M, Tw2j)

            # 点从世界转到相机坐标系
            pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
            pixel_coord_camj = np.matmul(K, pts_coord_camj)

            valid_idxj = np.where(np.logical_and.reduce((
                pixel_coord_camj[2] > 0,
                pixel_coord_camj[0]/pixel_coord_camj[2] >= 0,
                pixel_coord_camj[0]/pixel_coord_camj[2] <= W-1,
                pixel_coord_camj[1]/pixel_coord_camj[2] >= 0,
                pixel_coord_camj[1]/pixel_coord_camj[2] <= H-1
            )))[0]

            if len(valid_idxj) == 0:
                continue

            pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
            pixel_coord_camj = pixel_coord_camj[:2, valid_idxj] / pixel_coord_camj[-1:, valid_idxj]
            round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)

            # 生成图像
            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            grid = np.stack((x,y), axis=-1).reshape(-1,2)

            imagej = interp_grid(pixel_coord_camj.T, pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(H,W,3)
            imagej = edgemask[...,None]*imagej + (1-edgemask[...,None])*np.pad(imagej[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

            # 生成 mask
            maskj = np.zeros((H,W), dtype=np.float32)
            maskj[round_coord_camj[1], round_coord_camj[0]] = 1
            maskj = maximum_filter(maskj, size=(9,9), axes=(0,1))
            imagej = maskj[...,None]*imagej + (1-maskj[...,None])*(-1)

            maskj = minimum_filter((imagej.sum(-1) != -3)*1, size=(11,11), axes=(0,1))
            imagej = maskj[...,None]*imagej + (1-maskj[...,None])*0

            # 保存图像
            img = Image.fromarray(np.round(imagej*255.).astype(np.uint8))
            output_path = f"ablation/cam_{idx}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            img2 = imgs[idx]
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            output_path_2 = f"debug/output/gt_{idx}.png"
            os.makedirs(os.path.dirname(output_path_2), exist_ok=True)
            img2.save(output_path_2)

            # 存入训练数据
            traindata['frames'].append({
                'image': img2.rotate(180),
                'transform_matrix': Pc2w.tolist(),
            })

        if self.for_gradio:
            progress(1, desc='[3/4] Finished.')

        return traindata
