import gc
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from datetime import datetime
from copy import deepcopy
import json
import os
import re
import subprocess
from pytorch3d.structures import Pointclouds
from torchvision.transforms import ToPILImage
import uuid
import shutil
import torchvision

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, DPMSolverMultistepScheduler
import sys
sys.path.append('midas_module')
from midas_module.midas.model_loader import load_model
import torch.nn.functional as F

from models.models import KeyframeGen, KeyframeInterp, save_point_cloud_as_ply, initialize_gs_camera_from_pytorch3d, get_c2w_matrix, c2w_to_w2c
from util.finetune_utils import finetune_depth_model, finetune_decoder
from util.chatGPT4 import TextpromptGen
from util.general_utils import apply_depth_colormap, save_video
from util.utils import save_depth_map, prepare_scheduler
from util.utils import load_example_yaml, merge_frames, merge_keyframes
from util.segment_utils import create_mask_generator
from pytorch3d.renderer.cameras import try_get_projection_transform

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)

from models.GSRefiner.arguments import GSParams, CameraParams
from models.GSRefiner.GSRefiner import GaussSplatRefiner

from scipy.spatial.transform import Rotation as R, Slerp

def interpolate_w2c_fixed_length(w2c_matrices, start_index):
    
    total_len = len(w2c_matrices)
    assert 0 <= start_index < total_len - 1, "start_index must be within the range of w2c_matrices length"

    w2c_start = w2c_matrices[start_index]s
    w2c_end = w2c_matrices[-1]
    interp_len = total_len - start_index

    R0 = w2c_start[:3, :3]
    R1 = w2c_end[:3, :3]
    t0 = w2c_start[:3, 3]
    t1 = w2c_end[:3, 3]

    rotations = R.from_matrix([R0, R1])
    slerp = Slerp([0, 1], rotations)

    times = np.linspace(0, 1, interp_len)
    interp_rots = slerp(times)
    interp_trans = np.outer(1 - times, t0) + np.outer(times, t1)

    interpolated = []
    for i in range(interp_len):
        mat = np.eye(4)
        mat[:3, :3] = interp_rots[i].as_matrix()
        mat[:3, 3] = interp_trans[i]
        interpolated.append(mat)

    return interpolated

def render_opengl_pytorch3d_to_file(
    points_world,          
    camera,                
    image_size=512,        
    point_colors=None,    
    output_path="render.png" 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(points_world, np.ndarray):
        points_world = torch.from_numpy(points_world).float()

    if point_colors is not None and isinstance(point_colors, np.ndarray):
        point_colors = torch.from_numpy(point_colors).float()

    points_world = points_world.to(device)
    if point_colors is not None:
        point_colors = point_colors.to(device)

    device = points_world.device
    N = points_world.shape[0]
    points_homo = torch.cat([points_world, torch.ones(N, 1, device=device)], dim=1)  # (N, 4)

    # 提取视图 & 投影矩阵
    W2C = camera.get_world_to_view_transform().get_matrix()[0]
    P = try_get_projection_transform(camera, {}).get_matrix()[0]
    M = P @ W2C

    # 世界 → 裁剪空间
    points_clip = (M @ points_homo.T).T  # (N, 4)
    points_ndc = points_clip[:, :3] / points_clip[:, 3:4]  # 透视除法

    # NDC → 像素
    u = ((points_ndc[:, 0] + 1.0) * 0.5 * image_size).clamp(0, image_size - 1)
    v = ((1.0 - points_ndc[:, 1]) * 0.5 * image_size).clamp(0, image_size - 1)

    # 可见性过滤
    valid = (points_ndc[:, 2] >= -1) & (points_ndc[:, 2] <= 1)
    u = u[valid]
    v = v[valid]
    if point_colors is not None:
        colors = point_colors[valid].detach().cpu().numpy()
    else:
        colors = np.ones((valid.sum().item(), 3), dtype=np.float32)

    # 构造图像
    img = np.zeros((image_size, image_size, 3), dtype=np.float32)
    u_int = u.detach().cpu().numpy().astype(np.int32)
    v_int = v.detach().cpu().numpy().astype(np.int32)
    img[v_int, u_int] = colors

    # 保存为 PNG
    img_uint8 = (img * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(img_uint8).save(output_path)
    print(f"[INFO] Saved rendered image to {os.path.abspath(output_path)}")

    return img_uint8

def evaluate(model):
    fps = model.config["save_fps"]
    save_root = Path(model.run_dir)

    video = (255 * torch.cat(model.images, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (255 * torch.cat(model.images[::-1], dim=0)).to(torch.uint8).detach().cpu()

    # 保存反向视频每一帧
    reverse_frame_dir = Path("ablation")
    reverse_frame_dir.mkdir(exist_ok=True)
    for idx, frame in enumerate(video_reverse):
        torchvision.utils.save_image(frame.float() / 255.0, reverse_frame_dir / f"frame_{idx:04d}.png")

    save_video(video, save_root / "output.mp4", fps=fps)
    save_video(video_reverse, save_root / "output_reverse.mp4", fps=fps)

def save_pil_images(image_list, save_dir, prefix='image', format='PNG'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx, img in enumerate(image_list):
        if isinstance(img, Image.Image):
            filename = f"{prefix}_{idx}.{format.lower()}"
            path = os.path.join(save_dir, filename)
            img.save(path, format=format)
        else:
            print(f"跳过第 {idx} 项：不是一个 PIL.Image 对象")

def save_temp_images(img_list, folder_lq):
    os.makedirs(folder_lq, exist_ok=True)
    filenames = []
    for i, img in enumerate(img_list):
        filename = f"{i}.png"  # 按照顺序命名：0.jpg, 1.jpg, ...
        filepath = os.path.join(folder_lq, filename)
        img.save(filepath, format='PNG')
        filenames.append(filename)
    return filenames

def call_swinir_jpeg_car(folder_lq, model_path, jpeg_q=10):

    cmd = [
        "python", "SwinIR/main_test_swinir.py",
        "--task", "color_jpeg_car",
        "--jpeg", str(jpeg_q),
        "--model_path", model_path,
        "--folder_gt", folder_lq,
    ]
    print("Running SwinIR command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def load_restored_images(folder_out, filenames):
    output_images = []
    for fname in filenames:
        base_name = os.path.splitext(fname)[0]  # 提取文件名（如 "0"）
        restored_name = f"{base_name}_SwinIR.png"
        out_path = os.path.join(folder_out, restored_name)

        if os.path.exists(out_path):
            img = Image.open(out_path).convert("RGB")
            output_images.append(img)
        else:
            print(f"⚠️ Restored file not found: {out_path}")
            output_images.append(None)  # 保持顺序一致，可选
    return output_images

def jpeg_artifact_removal(img_list, model_path="SwinIR/model_zoo/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth"):
    folder_lq = "temp_input"
    folder_out = "results/swinir_color_jpeg_car_jpeg10"

    #try:
    filenames = save_temp_images(img_list, folder_lq)

    call_swinir_jpeg_car(folder_lq, model_path)

    restored_images = load_restored_images(folder_out, filenames)

    # finally:
    #     shutil.rmtree(folder_lq, ignore_errors=True)
    #     shutil.rmtree(folder_out, ignore_errors=True)

    return restored_images

def evaluate_epoch(model, epoch, vmax=None):
    rendered_depth = model.rendered_depths[epoch].clamp(0).cpu().numpy()
    depth = model.depths[epoch].clamp(0).cpu().numpy()
    save_root = Path(model.run_dir) / "images"
    save_root.mkdir(exist_ok=True, parents=True)
    (save_root / "inpaint_input_image").mkdir(exist_ok=True, parents=True)
    (save_root / "frames").mkdir(exist_ok=True, parents=True)
    (save_root / "masks").mkdir(exist_ok=True, parents=True)
    (save_root / "post_masks").mkdir(exist_ok=True, parents=True)
    (save_root / "rendered_images").mkdir(exist_ok=True, parents=True)
    (save_root / "rendered_depths").mkdir(exist_ok=True, parents=True)
    (save_root / "depth").mkdir(exist_ok=True, parents=True)

    model.inpaint_input_image[epoch].save(save_root / "inpaint_input_image" / f"{epoch}.png")
    ToPILImage()(model.images[epoch][0]).save(save_root / "frames" / f"{epoch}.png")
    ToPILImage()(model.masks[epoch][0]).save(save_root / "masks" / f"{epoch}.png")
    ToPILImage()(model.post_masks[epoch][0]).save(save_root / "post_masks" / f"{epoch}.png")
    ToPILImage()(model.rendered_images[epoch][0]).save(save_root / "rendered_images" / f"{epoch}.png")
    save_depth_map(rendered_depth, save_root / "rendered_depths" / f"{epoch}.png", vmax=vmax)
    save_depth_map(depth, save_root / "depth" / f"{epoch}.png", vmax=vmax, save_clean=True)

    if hasattr(model, "outter_masks"):
        (save_root / "outter_masks").mkdir(exist_ok=True, parents=True)
        ToPILImage()(model.outter_masks[epoch]).save(save_root / "outter_masks" / f"{epoch}.png")
    if epoch == 0:
        with open(Path(model.run_dir) / "config.yaml", "w") as f:
            OmegaConf.save(model.config, f)


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")

def scale_w2c_matrix(w2c: np.ndarray, k: float) -> np.ndarray:

    R = w2c[:3, :3]
    T = w2c[:3, 3]
    
    w2c_scaled = np.eye(4)
    w2c_scaled[:3, :3] = R
    w2c_scaled[:3, 3] = k * T
    
    return w2c_scaled

class RecursiveSceneGenerator:
    def __init__(self, config, control_text, style_prompt, pt_gen, start_keyframe, scene_dict, rotation_path, cutoff_depth,
                           mask_generator, segment_model, segment_processor,
                           inpainter_pipeline, vae, vmax, kfgen_save_folder,
                           inpainting_resolution_gen, adaptive_negative_prompt, rollback_steps=1, max_feedback_rounds=25):
        
        self.prompt_list = []
        self.rollback_steps = rollback_steps
        self.max_feedback_rounds = max_feedback_rounds
        self.all_kf = []
        self.all_keyframes_data = []
        self.config = config
        self.control_text = control_text
        self.style_prompt = style_prompt
        self.pt_gen = pt_gen
        self.scene_dict = scene_dict
        self.start_keyframe = start_keyframe
        self.rotation_path = rotation_path
        self.cutoff_depth = cutoff_depth
        self.mask_generator = mask_generator
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.inpainter_pipeline = inpainter_pipeline
        self.vae = vae
        self.vmax = vmax
        self.kfgen_save_folder = kfgen_save_folder
        self.inpainting_resolution_gen = inpainting_resolution_gen
        self.adaptive_negative_prompt = adaptive_negative_prompt
        self.dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        self.save_dir = Path(config['runs_dir']) / f"{self.dt_string}"
        self.kf_gens = []
        self.scene_prompt = []
        self.all_render = []
        self.mis = -1
        for i in range (config['num_scenes']):
            if self.config['use_gpt']:
                control_text_this = self.control_text[i] if isinstance(self.control_text, list) else None
                scene_dict = self.pt_gen.run_conversation(
                    scene_name=self.scene_dict['scene_name'],
                    entities=self.scene_dict['entities'],
                    style=self.style_prompt,
                    background=self.scene_dict['background'],
                    control_text=control_text_this
                )

            inpainting_prompt = self.pt_gen.generate_prompt(
                style=self.style_prompt,
                entities=scene_dict['entities'],
                background=scene_dict['background'],
                scene_name=scene_dict['scene_name']
            )
            self.scene_prompt.append(inpainting_prompt)

    def generate(self):
        self.scene_list = self._generate_recursive(t=0, feedback_round=0)
        os.makedirs(self.save_dir, exist_ok=True)
        for i in range(len(self.all_kf)):
            self.all_kf[i].save(self.save_dir / f"keyframe_{i}.png")
        return self.scene_list

    def _generate_recursive(self, t, feedback_round, issues=''):
        """
        Internal recursive generation function.

        Args:
            t (int): Current frame index.
            feedback_round (int): Current feedback recursion depth.

        Returns:
            List: Updated scene list.
        """
        scene_id = t // self.config['num_keyframes']
        kf_id = t % self.config['num_keyframes']
        flag = False
        if t > (self.config['num_keyframes']) * (self.config['num_scenes']) - 1:
            print(t)
            print((self.config['num_keyframes']) * (self.config['num_scenes']) - 1)
            print("Scene generation completed.")
            return self.all_keyframes_data, self.all_kf, self.save_dir

        if feedback_round > self.max_feedback_rounds:
            print(f"Max feedback rounds reached at frame {t}. Skipping further repair.")
            flag = True

        # kf_gen = self.kf_gens[t]
        # if kf_gen is None:
        if t == 0:
            start_kf = self.start_keyframe
        else:
            start_kf = ToPILImage()(self.kf_gens[t - 1]['kf2_image'][0])
        
        kf_gen = self.generate_kf_gen(scene_id, kf_id, start_kf)

        print(f"Generated frame: scene {scene_id}, keyframe {kf_id}, frame {t}")

        if t < len(self.kf_gens):
            self.kf_gens[t] = kf_gen
        else:
            self.kf_gens.append(kf_gen)

        iss, res = self.detect_scene_issue(kf_gen['kf2_image'][0])
        if iss and flag==False:
            print(f"Issues at frame {t}: {iss}")
            rollback_start = max(0, t - self.rollback_steps)
            scene_id_back = rollback_start // self.config['num_keyframes']
            kf_id_back = rollback_start % self.config['num_keyframes']
            print(f"Rolling back to scene {scene_id_back}, keyframe{kf_id_back}, frame {rollback_start}")

            for i in range(rollback_start, t+1):
                if i <= self.mis:
                    if rollback_start != t:
                        rollback_start = i + 1
                    continue
                issues_i, mask, new_prompt = self.detect_same_issue(self.kf_gens[i]['kf2_image'][0], res)
                if issues_i:
                    print(f"modifying frame {i}, Rewriting at frame {i+1}")
                    scene_id = i // self.config['num_keyframes']
                    kf_id = i % self.config['num_keyframes']
                    self.mis = i
                    if i == 0:
                        start_kf = self.start_keyframe
                    else:
                        start_kf = ToPILImage()(self.kf_gens[i - 1]['kf2_image'][0])

                    os.makedirs(self.save_dir, exist_ok=True)
                    self.all_kf[i].save(self.save_dir / f"modify_{i}.png")
                    self.modify(scene_id, kf_id, start_kf=start_kf, original_image=self.all_render[i], mask=mask, modify_prompt=new_prompt)
                    self.all_kf[i].save(self.save_dir / f"modified_{i}.png")
                    break
                else:
                    if rollback_start != t:
                        rollback_start = i + 1
            
            self.kf_gens = self.kf_gens[:rollback_start+1]
            self.all_kf = self.all_kf[:rollback_start+1]
            self.all_render = self.all_render[:rollback_start+1]
            self.all_keyframes_data = self.all_keyframes_data[:rollback_start+1]

            return self._generate_recursive(
                t=rollback_start + 1,
                feedback_round=feedback_round + 1,
                issues=iss
            )
        else:
            return self._generate_recursive(t + 1, feedback_round, issues=issues)

    def modify(self, scene_idx, kf_id, start_kf, original_image=None, mask=None, modify_prompt=""):
        """
        Modify the generated image based on detected issues.
        """
        seeding(-1)
        if original_image is None:
            original_image = self.kf_gens[scene_idx]['kf2_image'][0]
        if mask is None:
            mask = self.kf_gens[scene_idx]['kf2_mask'][0]

        id = scene_idx * self.config['num_keyframes'] + kf_id
        inpainting_prompt = self.scene_prompt[scene_idx]

        rotation = self.rotation_path[scene_idx * self.config['num_keyframes'] + kf_id]
        if self.config['skip_gen']:
                kf_gen_dict = torch.load(self.kfgen_save_folder / f"s{scene_idx:02d}_k{kf_id:01d}_gen_dict.pt")
                kf1_depth, kf2_depth = kf_gen_dict['kf1_depth'], kf_gen_dict['kf2_depth']
                kf1_image, kf2_image = kf_gen_dict['kf1_image'], kf_gen_dict['kf2_image']
                kf1_camera, kf2_camera = kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera']
                kf2_mask = kf_gen_dict['kf2_mask']
                inpainting_prompt = kf_gen_dict['inpainting_prompt']
                self.adaptive_negative_prompt = kf_gen_dict['adaptive_negative_prompt']
        else:
             # 否则，执行关键帧生成流程
                regen_negative_prompt = ""
                self.config['inpainting_resolution_gen'] = self.inpainting_resolution_gen

                for regen_id in range(self.config['regenerate_times'] + 1):
                    if regen_id > 0:
                        seeding(1)
                    depth_model, _, _, _ = load_model(
                        torch.device("cuda"), 'dpt_beit_large_512.pt', 
                        'dpt_beit_large_512', optimize=False
                    )
                    kf_gen = KeyframeGen(
                        self.config, self.inpainter_pipeline, self.mask_generator, depth_model, self.vae, rotation,
                        start_kf, modify_prompt, regen_negative_prompt + self.adaptive_negative_prompt,
                        segment_model=self.segment_model, segment_processor=self.segment_processor
                    ).to(self.config["device"])
                    save_root = Path(kf_gen.run_dir) / "images"
                    kf_idx = 0

                    # 1. 处理第一个关键帧
                    save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(),
                                   save_root / 'kf1_original', vmin=0, vmax=self.vmax)
                    kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=self.cutoff_depth)
                    save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(),
                                   save_root / 'kf1_processed', vmin=0, vmax=self.vmax)
                    evaluate_epoch(kf_gen, kf_idx, vmax=self.vmax)

                    # 2. 处理第二个关键帧
                    kf_idx = 1
                    render_output = kf_gen.render(kf_idx, use_gaussian=False)

                    mask = mask.unsqueeze(0).unsqueeze(0)
                    mask_and = mask.to(render_output["inpaint_mask"].device) * (render_output["inpaint_mask"])
                    num_ones = (mask_and == 1).sum().item()
                    print(f"Number of pixels needing inpainting: {num_ones}")

                    print(original_image.shape)
                    # 将 mask 移动到图像的设备
                    # mask_and = mask
                    mask_and = mask_and.to(original_image.device)
                    

                    # 扩展 mask 到三通道
                    mask_3ch = mask_and.expand(-1, 3, -1, -1)  # 形状 (1, 3, 512, 512)

                    # 归一化图像（如果图像是 uint8）
                    if original_image.dtype == torch.uint8:
                        original_image = original_image.float() / 255.0

                    # 构造白图
                    white_image = torch.ones_like(original_image)

                    # 应用遮罩：mask=1 的地方为白，其它为原图
                    original_image = original_image * (1 - mask_3ch) + white_image * mask_3ch
                    to_pil = ToPILImage()

                    img_tensor = original_image[0].clamp(0, 1).cpu()  # 保证范围在 [0, 1] 且在 CPU 上

                    # 转换为 PIL 图像
                    img_pil = to_pil(img_tensor)

                    # 保存
                    img_pil.save(self.save_dir / f"original_image_masked_{id}.png")
                    print("Image saved to original_image_masked.png")


                    inpaint_output = kf_gen.inpaint(original_image, 
                                                    mask_and)
                    regenerate = False

                    if not regenerate:
                        # 如果无需重新生成，则直接退出该循环
                        break

                    # 回收显存
                    depth_model = kf_gen.depth_model.to('cpu')
                    kf_gen.depth_model = None
                    del depth_model
                    empty_cache()
                
                 # finetune decoder
                # if self.config["finetune_decoder_gen"]:
                #     ToPILImage()(inpaint_output["inpainted_image"].detach()[0]).save(save_root / 'kf2_before_ft.png')
                #     finetune_decoder(self.config, kf_gen, render_output, inpaint_output, 
                #                      self.config['num_finetune_decoder_steps'])
                
                kf_gen.update_images_and_masks(inpaint_output["latent"], render_output["inpaint_mask"])

                kf2_depth_should_be = render_output['rendered_depth']
                mask_to_align_depth = ~(render_output["inpaint_mask_512"]>0) & \
                                      (kf2_depth_should_be < self.cutoff_depth + kf_gen.kf_delta_t)
                mask_to_cutoff_depth = ~(render_output["inpaint_mask_512"]>0) & \
                                       (kf2_depth_should_be >= self.cutoff_depth + kf_gen.kf_delta_t)

                if self.config["finetune_depth_model"]:
                    finetune_depth_model(
                        self.config, kf_gen, kf2_depth_should_be, kf_idx,
                        mask_align=mask_to_align_depth,
                        mask_cutoff=mask_to_cutoff_depth,
                        cutoff_depth=self.cutoff_depth + kf_gen.kf_delta_t
                    )
                with torch.no_grad():
                    kf2_ft_depth_original, kf2_ft_disp_original = kf_gen.get_depth(kf_gen.images[kf_idx])
                    kf_gen.depths.append(kf2_ft_depth_original)
                    kf_gen.disparities.append(kf2_ft_disp_original)

                depth_model = kf_gen.depth_model.to('cpu')
                kf_gen.depth_model = None
                del depth_model
                empty_cache()

                kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=self.cutoff_depth + kf_gen.kf_delta_t)
                save_depth_map(kf_gen.depths[-1].cpu().numpy(), save_root / 'kf2_ft_depth_processed',
                               vmin=0, vmax=self.vmax)

                kf_gen.vae.decoder = deepcopy(kf_gen.decoder_copy)
                evaluate_epoch(kf_gen, kf_idx, vmax=self.vmax)

                kf1_depth, kf2_depth = kf_gen.depths[0], kf_gen.depths[-1]
                kf1_image, kf2_image = kf_gen.images[0], kf_gen.images[1]
                kf1_camera, kf2_camera = kf_gen.cameras[0], kf_gen.cameras[1]
                kf2_mask = render_output["inpaint_mask_512"]
                kf_gen_dict = {
                    'kf1_depth': kf1_depth, 'kf2_depth': kf2_depth,
                    'kf1_image': kf1_image, 'kf2_image': kf2_image,
                    'kf1_camera': kf1_camera, 'kf2_camera': kf2_camera,
                    'kf2_mask': kf2_mask, 'inpainting_prompt': inpainting_prompt,
                    'adaptive_negative_prompt': self.adaptive_negative_prompt,
                    'rotation': rotation
                }

                torch.save(kf_gen_dict, self.kfgen_save_folder / f"s{scene_idx:02d}_k{kf_id:01d}_gen_dict.pt")

                self.all_kf[id] = ToPILImage()(kf_gen.images[1][0])
                self.all_render[id] = kf_gen.images[1]
                self.kf_gens[id] = kf_gen_dict
                self.all_keyframes_data[id] = {
                "scene_idx": scene_idx,
                "keyframe_idx": kf_id,
                "rotation": rotation,
                "inpainting_prompt": inpainting_prompt,
                "adaptive_negative_prompt": self.adaptive_negative_prompt
                }
        seeding(1)

        print(f"[Keyframe modified: scene {scene_idx} keyframe {kf_id}]")
        return kf_gen_dict

    def generate_kf_gen(self, scene_idx, kf_id, start_kf, modify_prompt=""):
        id = scene_idx * self.config['num_keyframes'] + kf_id
        

        inpainting_prompt = self.scene_prompt[scene_idx]

        rotation = self.rotation_path[scene_idx * self.config['num_keyframes'] + kf_id]
        if self.config['skip_gen']:
                kf_gen_dict = torch.load(self.kfgen_save_folder / f"s{scene_idx:02d}_k{kf_id:01d}_gen_dict.pt")
                kf1_depth, kf2_depth = kf_gen_dict['kf1_depth'], kf_gen_dict['kf2_depth']
                kf1_image, kf2_image = kf_gen_dict['kf1_image'], kf_gen_dict['kf2_image']
                kf1_camera, kf2_camera = kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera']
                kf2_mask = kf_gen_dict['kf2_mask']
                inpainting_prompt = kf_gen_dict['inpainting_prompt']
                self.adaptive_negative_prompt = kf_gen_dict['adaptive_negative_prompt']
        else:
             # 否则，执行关键帧生成流程
                regen_negative_prompt = ""
                self.config['inpainting_resolution_gen'] = self.inpainting_resolution_gen

                for regen_id in range(self.config['regenerate_times'] + 1):
                    if regen_id > 0:
                        seeding(1)
                    depth_model, _, _, _ = load_model(
                        torch.device("cuda"), 'dpt_beit_large_512.pt', 
                        'dpt_beit_large_512', optimize=False
                    )
                    kf_gen = KeyframeGen(
                        self.config, self.inpainter_pipeline, self.mask_generator, depth_model, self.vae, rotation,
                        start_kf, inpainting_prompt, regen_negative_prompt + self.adaptive_negative_prompt + modify_prompt,
                        segment_model=self.segment_model, segment_processor=self.segment_processor
                    ).to(self.config["device"])
                    save_root = Path(kf_gen.run_dir) / "images"
                    kf_idx = 0

                    save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(),
                                   save_root / 'kf1_original', vmin=0, vmax=self.vmax)
                    kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=self.cutoff_depth)
                    save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(),
                                   save_root / 'kf1_processed', vmin=0, vmax=self.vmax)
                    evaluate_epoch(kf_gen, kf_idx, vmax=self.vmax)

                    kf_idx = 1
                    render_output = kf_gen.render(kf_idx, use_gaussian=False)
                    inpaint_output = kf_gen.inpaint(render_output["rendered_image"], 
                                                    render_output["inpaint_mask"])

                    regenerate_information = {}
                    if self.config['enable_regenerate'] and regen_id <= self.config['regenerate_times'] -1:
                        gpt_border, gpt_blur = self.pt_gen.evaluate_image(
                            ToPILImage()(inpaint_output['inpainted_image'][0]),
                            eval_blur=False
                        )
                        regenerate_information['gpt_border'] = gpt_border
                        regenerate_information['gpt_blur'] = gpt_blur

                        if gpt_border:
                            print("chatGPT-4 says the image has border!")
                            regen_negative_prompt += "border, "
                        if gpt_blur:
                            print("chatGPT-4 says the image has blurry effect!")
                            regen_negative_prompt += "blur, "

                        regenerate = gpt_border
                    else:
                        regenerate = False

                    with open(save_root / 'regenerate_info.json', 'w') as json_file:
                        json.dump(regenerate_information, json_file, indent=4)

                    if not regenerate:
                        break
                    if regen_id >= self.config['regenerate_times'] -1:
                        print("Regenerating faild after {} times".format(self.config['regenerate_times']))
                        if gpt_border:
                            print("Use crop to solve border problem!")
                            self.config['inpainting_resolution_gen'] = 560
                        else:
                            break

                    depth_model = kf_gen.depth_model.to('cpu')
                    kf_gen.depth_model = None
                    del depth_model
                    empty_cache()

                if self.config["finetune_decoder_gen"]:
                    ToPILImage()(inpaint_output["inpainted_image"].detach()[0]).save(save_root / 'kf2_before_ft.png')
                    finetune_decoder(self.config, kf_gen, render_output, inpaint_output, 
                                     self.config['num_finetune_decoder_steps'])
                
                kf_gen.update_images_and_masks(inpaint_output["latent"], render_output["inpaint_mask"])

                kf2_depth_should_be = render_output['rendered_depth']
                mask_to_align_depth = ~(render_output["inpaint_mask_512"]>0) & \
                                      (kf2_depth_should_be < self.cutoff_depth + kf_gen.kf_delta_t)
                mask_to_cutoff_depth = ~(render_output["inpaint_mask_512"]>0) & \
                                       (kf2_depth_should_be >= self.cutoff_depth + kf_gen.kf_delta_t)

                if self.config["finetune_depth_model"]:
                    finetune_depth_model(
                        self.config, kf_gen, kf2_depth_should_be, kf_idx,
                        mask_align=mask_to_align_depth,
                        mask_cutoff=mask_to_cutoff_depth,
                        cutoff_depth=self.cutoff_depth + kf_gen.kf_delta_t
                    )
                with torch.no_grad():
                    kf2_ft_depth_original, kf2_ft_disp_original = kf_gen.get_depth(kf_gen.images[kf_idx])
                    kf_gen.depths.append(kf2_ft_depth_original)
                    kf_gen.disparities.append(kf2_ft_disp_original)

                depth_model = kf_gen.depth_model.to('cpu')
                kf_gen.depth_model = None
                del depth_model
                empty_cache()

                kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=self.cutoff_depth + kf_gen.kf_delta_t)
                save_depth_map(kf_gen.depths[-1].cpu().numpy(), save_root / 'kf2_ft_depth_processed',
                               vmin=0, vmax=self.vmax)

                kf_gen.vae.decoder = deepcopy(kf_gen.decoder_copy)
                evaluate_epoch(kf_gen, kf_idx, vmax=self.vmax)

                kf1_depth, kf2_depth = kf_gen.depths[0], kf_gen.depths[-1]
                kf1_image, kf2_image = kf_gen.images[0], kf_gen.images[1]
                kf1_camera, kf2_camera = kf_gen.cameras[0], kf_gen.cameras[1]
                kf2_mask = render_output["inpaint_mask_512"]
                kf_gen_dict = {
                    'kf1_depth': kf1_depth, 'kf2_depth': kf2_depth,
                    'kf1_image': kf1_image, 'kf2_image': kf2_image,
                    'kf1_camera': kf1_camera, 'kf2_camera': kf2_camera,
                    'kf2_mask': kf2_mask, 'inpainting_prompt': inpainting_prompt,
                    'adaptive_negative_prompt': self.adaptive_negative_prompt,
                    'rotation': rotation
                }

                torch.save(kf_gen_dict, self.kfgen_save_folder / f"s{scene_idx:02d}_k{kf_id:01d}_gen_dict.pt")

                self.all_kf.append(ToPILImage()(kf_gen.images[1][0]))
                self.all_render.append(kf_gen.images[1])
                self.all_keyframes_data.append({
                "scene_idx": scene_idx,
                "keyframe_idx": kf_id,
                "rotation": rotation,
                "inpainting_prompt": inpainting_prompt,
                "adaptive_negative_prompt": self.adaptive_negative_prompt
                })


        print(f"[Keyframe generated: scene {scene_idx} keyframe {kf_id}]")
        return kf_gen_dict

    def detect_scene_issue(self, img):
        has_issue, issue_description = self.pt_gen.detect_iss(ToPILImage()(img))
        if has_issue:
            print(f"Detected issues: {issue_description}")
        else:
            print("No issues detected.")
        return has_issue, issue_description 

    def detect_same_issue(self, img, iss_des):
        has_issue, window = self.pt_gen.check_image_issue_with_window(ToPILImage()(img), iss_des)
        if has_issue:
            mask = torch.zeros((512, 512), dtype=torch.float32)
            print(window)
            mask[window['y_min']:window['y_max'], window['x_min']:window['x_max']] = 1.0
            new_prompt = window['prompt']
            return has_issue, mask, new_prompt
        else:
            mask = torch.zeros((512, 512), dtype=torch.float32)
            new_prompt = " "
            print("No issues detected.")
            return has_issue, mask, new_prompt

def interpolate_all_keyframes(config, all_keyframes_data, save_dir, rotation_path, cutoff_depth,
                              vmax, inpainter_pipeline, vae, adaptive_negative_prompt, kfgen_save_folder):
    all_rundir = []

    for keyframe_info in all_keyframes_data:
        i = keyframe_info["scene_idx"]
        j = keyframe_info["keyframe_idx"]
        rotation = keyframe_info["rotation"]
        inpainting_prompt = keyframe_info["inpainting_prompt"]

        if config['skip_interp']:
            continue
        kf_gen_dict = torch.load(kfgen_save_folder / f"s{i:02d}_k{j:01d}_gen_dict.pt")
        kf1_depth, kf2_depth = kf_gen_dict['kf1_depth'], kf_gen_dict['kf2_depth']
        kf1_image, kf2_image = kf_gen_dict['kf1_image'], kf_gen_dict['kf2_image']
        kf1_camera, kf2_camera = kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera']
        kf2_mask = kf_gen_dict['kf2_mask']
        
        is_last_scene = (i == config['num_scenes'] - 1)
        is_last_keyframe = (j == config['num_keyframes'] - 1)
        try:
            is_next_rotation = rotation_path[i*config['num_keyframes'] + j + 1] != 0
        except IndexError:
            is_next_rotation = False
        try:
            is_previous_rotation = rotation_path[i*config['num_keyframes'] + j - 1] != 0
        except IndexError:
            is_previous_rotation = False

        is_beginning = (i == 0 and j == 0)
        speed_up = (rotation == 0) and ((is_last_scene and is_last_keyframe) or is_next_rotation)
        speed_down = (rotation == 0) and (is_beginning or is_previous_rotation)

        total_frames = config["frames"]
        if speed_up:
            total_frames += config["frames"] // 5
        if speed_down:
            total_frames += config["frames"] // 5

        kf_interp = KeyframeInterp(
            config, inpainter_pipeline, None, vae, rotation, 
            ToPILImage()(kf1_image[0]), inpainting_prompt, adaptive_negative_prompt,
            kf2_upsample_coef=config['kf2_upsample_coef'],
            kf1_image=kf1_image, kf2_image=kf2_image,
            kf1_depth=kf1_depth, kf2_depth=kf2_depth,
            kf1_camera=kf1_camera, kf2_camera=kf2_camera, kf2_mask=kf2_mask,
            speed_up=speed_up, speed_down=speed_down, total_frames=total_frames
        ).to(config["device"])

        save_root = Path(kf_interp.run_dir) / "images"
        save_root.mkdir(exist_ok=True, parents=True)
        ToPILImage()(kf1_image[0]).save(save_root / "kf1.png")
        ToPILImage()(kf2_image[0]).save(save_root / "kf2.png")

        kf2_camera_upsample, kf2_depth_upsample, kf2_mask_upsample, kf2_image_upsample = kf_interp.upsample_kf2()

        kf_interp.update_additional_point_cloud(
            kf2_depth_upsample, kf2_image_upsample, valid_mask=kf2_mask_upsample,
            camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2
        )
        inconsistent_additional_point_index = kf_interp.visibility_check()
        kf2_depth_updated = kf_interp.update_additional_point_depth(
            inconsistent_additional_point_index, depth=kf2_depth_upsample,
            mask=kf2_mask_upsample
        )
        kf_interp.reset_additional_point_cloud()
        kf_interp.update_additional_point_cloud(
            kf2_depth_updated, kf2_image_upsample, valid_mask=kf2_mask_upsample,
            camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2
        )

        kf_interp.depths[0] = F.interpolate(kf2_depth_updated, size=(512, 512), mode="nearest")
        evaluate_epoch(kf_interp, 0, vmax=vmax)

        for epoch in tqdm(range(1, total_frames + 1)):
            render_output_kf1 = kf_interp.render_kf1(epoch)

            inpaint_output = kf_interp.inpaint(
                render_output_kf1["rendered_image"], render_output_kf1["inpaint_mask"]
            )

            if config["finetune_decoder_interp"]:
                finetune_decoder(config, kf_interp, render_output_kf1, inpaint_output,
                                 config["num_finetune_decoder_steps_interp"])

            kf_interp.update_images_and_masks(inpaint_output["latent"], render_output_kf1["inpaint_mask"])

            kf_interp.update_additional_point_cloud(render_output_kf1["rendered_depth"], 
                                                    kf_interp.images[-1], append_depth=True)

            kf_interp.vae.decoder = deepcopy(kf_interp.decoder_copy)
            with torch.no_grad():
                kf_interp.images_orig_decoder.append(
                    kf_interp.decode_latents(inpaint_output["latent"]).detach()
                )

            evaluate_epoch(kf_interp, epoch, vmax=cutoff_depth*0.95)
            empty_cache()

        kf_interp.images.append(kf1_image)
        kf_interp.cameras.append(kf1_camera)

        save_point_cloud_as_ply(
            torch.cat([kf_interp.points_3d, kf_interp.additional_points_3d], dim=0)*500,
            kf_interp.run_dir / 'final_point_cloud.ply',
            torch.cat([kf_interp.kf1_colors, kf_interp.additional_colors], dim=0)
        )
        all_rundir.append(kf_interp.run_dir)

        if config['use_gs_refiner'] and config['rotation_path'][i*config['num_keyframes'] + j] == 0:
            my_points, my_colors = kf_interp.get_points_3d()
            my_points = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in my_points]
            my_points = np.array(my_points)
            print(my_points.shape)
            my_points[:, 0] = - my_points[:, 0]
            my_points = my_points * 1e4

            my_colors = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in my_colors]
            my_colors = np.array(my_colors)

            frames = []
            render_poses_list = []
            img_list = []
            w2c_list = []
            to_pil = ToPILImage()
            for epoch in range(1, total_frames):
                render_opengl_pytorch3d_to_file(my_points, kf_interp.cameras[epoch], image_size=512, point_colors=my_colors, output_path = f"{epoch}.png")
                c2w = get_c2w_matrix(kf_interp.cameras[epoch])
                c2w = c2w[0]
                image = kf_interp.images[epoch]
                pil_image = to_pil(image[0])
                w2c = np.linalg.inv(c2w)
                w2c_list.append(w2c)
                w2c = scale_w2c_matrix(w2c, 1e4)

                R = np.array(w2c[:3, :3])
                T = np.array(w2c[:3, 3:] * 1e4)
                # w2c[0, :] *= -1
                print(T.shape)
                R_c2w = R.T
                yz_reverse = np.diag([1, -1, -1])
                T_c2w_flipped = -np.matmul(R_c2w, np.matmul(yz_reverse, T))
                R_c2w_flipped = yz_reverse @ R_c2w

                Pc2w = np.concatenate([R_c2w_flipped, T_c2w_flipped], axis=1)
                Pc2w = np.vstack([Pc2w, [0, 0, 0, 1]])
                render_poses_list.append(w2c[:3, :])
                print(w2c[:3, :])
                print("next")
                img_list.append(pil_image)
                frames.append({"image": pil_image.rotate(180), "transform_matrix": Pc2w.tolist()})

            camerapath = np.array(render_poses_list)
            save_pil_images(img_list, './debug_img/output_images', prefix='frame', format='JPEG')
            debug_dir = "debug"
            frames_json = []
            for frame in frames:
                frame_data = {
                    "image_path": os.path.join(debug_dir, f"frame_{frames.index(frame)}.png"),
                    "transform_matrix": frame["transform_matrix"]
                }
                frames_json.append(frame_data)
                frame["image"].save(frame_data["image_path"])

            save_for_debug = True
            if save_for_debug == True:
                debug_dir = "debug"
                os.makedirs(debug_dir, exist_ok=True)
                np.save(os.path.join(debug_dir, "my_points.npy"), my_points)
                np.save(os.path.join(debug_dir, "my_colors.npy"), my_colors)

                with open(os.path.join(debug_dir, "frames.json"), "w") as f:
                    json.dump(frames_json, f, indent=4)

            example_name = f"{i*config['num_keyframes']+j}_frames"

            refiner = GaussSplatRefiner( 
                    for_gradio = False,
                    save_dir=save_dir
                )
            my_points = my_points.T
            refiner.create(
                camerapath = camerapath, 
                pts_coord_world = my_points, 
                pts_colors = my_colors,
                imgs = img_list
            )
            # refiner.create_from_frames(frames=frames, pc = my_points, pc_colors = my_colors)
            refiner.render_video(example_name = example_name, fps = config["save_fps"])

        else:
            if config['rotation_path'][i*config['num_keyframes'] + j] != 0:
                video = (255 * torch.cat(kf_interp.images, dim=0)).to(torch.uint8).detach().cpu()
                fps = kf_interp.config["save_fps"]
                example_name = f"{i*config['num_keyframes']+j}_frames"
                save_video(video, save_dir / f"{example_name}.mp4", fps=fps)
            else:
                evaluate(kf_interp)

    if config['use_gs_refiner'] == False:
        if not config['skip_interp']:
            merge_frames(all_rundir, save_dir=save_dir, fps=config["save_fps"],
                         is_forward=True, save_depth=False, save_gif=False)
    else:
        folder_path = save_dir
        output_video = os.path.join(save_dir, "merged_video.mp4")

        pattern = re.compile(r"(\d+)_frames\.mp4")
        video_files = []

        for file in os.listdir(folder_path):
            match = pattern.match(file)
            if match:
                idx_j = int(match.group(1))
                video_files.append((idx_j, os.path.join(folder_path, file)))

        video_files.sort(key=lambda x: x[0], reverse=True)
        video_files = [file_path for _, file_path in video_files]

        if video_files:
            list_file = "video_list.txt"
            with open(list_file, "w") as f:
                for video in video_files:
                    f.write(f"file '{video}'\n")
            command = f"ffmpeg -f concat -safe 0 -i {list_file} -c copy {output_video}"
            subprocess.run(command, shell=True)
            os.remove(list_file)
            print(f"finished {output_video}")
        else:
            print("not found")

    return

def main_loop(config, control_text, style_prompt, pt_gen, start_keyframe, scene_dict, rotation_path, cutoff_depth,
              mask_generator, segment_model, segment_processor,
              inpainter_pipeline, vae, vmax, kfgen_save_folder,
              inpainting_resolution_gen, adaptive_negative_prompt):
    
    recursive_generator = RecursiveSceneGenerator(
        config=config,
        control_text=control_text,
        style_prompt=style_prompt,
        pt_gen=pt_gen,
        start_keyframe=start_keyframe,
        scene_dict=scene_dict,
        rotation_path=rotation_path,
        cutoff_depth=cutoff_depth,
        mask_generator=mask_generator,
        segment_model=segment_model,
        segment_processor=segment_processor,
        inpainter_pipeline=inpainter_pipeline,
        vae=vae,
        vmax=vmax,
        kfgen_save_folder=kfgen_save_folder,
        inpainting_resolution_gen=inpainting_resolution_gen,
        adaptive_negative_prompt=adaptive_negative_prompt
    )

    all_keyframes_data, all_keyframes, save_dir = recursive_generator.generate()

    interpolate_all_keyframes(
        config, all_keyframes_data, save_dir, rotation_path, cutoff_depth,
        vmax, inpainter_pipeline, vae, adaptive_negative_prompt, kfgen_save_folder
    )

    pt_gen.write_all_content(save_dir=save_dir)

    return save_dir

def run(config):

    ###### ------------------ Load modules ------------------ ######

    if config['skip_gen']:
        kfgen_save_folder = Path(config['runs_dir']) / f"{config['kfgen_load_dt_string']}_kfgen"
    else:
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        kfgen_save_folder = Path(config['runs_dir']) / f"{dt_string}_kfgen"
    kfgen_save_folder.mkdir(exist_ok=True, parents=True)
    cutoff_depth = config['fg_depth_range'] + config['depth_shift']
    vmax = cutoff_depth * 2
    inpainting_resolution_gen = config['inpainting_resolution_gen']
    seeding(config["seed"])

    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

    mask_generator = create_mask_generator()

    # all_rundir = []
    from PIL import Image
    yaml_data = load_example_yaml(config["example_name"], 'examples/examples.yaml')
    start_keyframe = Image.open(yaml_data['image_filepath']).convert('RGB').resize((512, 512))
    content_prompt, style_prompt, adaptive_negative_prompt, background_prompt, control_text = yaml_data['content_prompt'], yaml_data['style_prompt'], yaml_data['negative_prompt'], yaml_data.get('background', None), yaml_data.get('control_text', None)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "
    all_keyframes = [start_keyframe]
    
    if isinstance(control_text, list):
        config['num_scenes'] = len(control_text)
    pt_gen = TextpromptGen(config['runs_dir'], isinstance(control_text, list))
    content_list = content_prompt.split(',')
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
    inpainting_prompt = style_prompt + ', ' + content_prompt

    inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config["stable_diffusion_checkpoint"],
            safety_checker=None,
            torch_dtype=torch.float16,
            revision="fp16",
        ).to(config["device"])
    inpainter_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(inpainter_pipeline.scheduler.config)
    inpainter_pipeline.scheduler = prepare_scheduler(inpainter_pipeline.scheduler)
    vae = AutoencoderKL.from_pretrained(config["stable_diffusion_checkpoint"], subfolder="vae").to(config["device"])

    rotation_path = config['rotation_path']
    assert len(rotation_path) >= config['num_scenes'] * config['num_keyframes']

    save_dir = main_loop(
        config, control_text, style_prompt, pt_gen, start_keyframe, scene_dict, rotation_path, cutoff_depth,
        mask_generator, segment_model, segment_processor,
        inpainter_pipeline, vae, vmax, kfgen_save_folder,
        inpainting_resolution_gen, adaptive_negative_prompt
    )

    import glob
    frames_path = save_dir
    image_paths = sorted(glob.glob(os.path.join(frames_path, "*.png")))
    frames = [Image.open(path) for path in image_paths]
    return frames

def generate(prompt, image_path):
    base_config = OmegaConf.load("./config/base-config.yaml")
    example_config = OmegaConf.load("./config/village.yaml")
    config = OmegaConf.merge(base_config, example_config)

    config['rotation_path'] = image_path
    
    POSTMORTEM = config['debug']
    if POSTMORTEM:
        try:
            frames = run(config)
        except Exception as e:
            print(e)
            import ipdb
            ipdb.post_mortem()
    else:
        frames = run(config)

    return frames

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config",
        default="./config/village.yaml"
    )
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    POSTMORTEM = config['debug']
    if POSTMORTEM:
        try:
            run(config)
        except Exception as e:
            print(e)
            import ipdb
            ipdb.post_mortem()
    else:
        run(config)

