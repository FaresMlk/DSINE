import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from .utils import utils
from typing import List
from pathlib import Path
import json
import trimesh
import sys
ROOT_ScanSDKCloud_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_ScanSDKCloud_path)
from modules.BodyMeasurements.src.measurements.regressor_meas import extra_measurements_model
from modules.smplx import smplx
#import smplx

def extract_intrinsicss(input_dir_path: Path) -> torch.Tensor:
    target_scan_info_path = Path(input_dir_path) / "frame_info.json"

    target_scan_info = json.loads(target_scan_info_path.read_text())
    fx = target_scan_info["intrinsics"]["matrix"][1][1]
    fy = target_scan_info["intrinsics"]["matrix"][0][0]
    px = target_scan_info["intrinsics"]["matrix"][2][1]
    py = target_scan_info["intrinsics"]["matrix"][2][0]
    image_height, image_width = target_scan_info["imageResolution"]

    fx_ndc = fx * 2.0 / min(image_width, image_height)
    fy_ndc = fy * 2.0 / min(image_width, image_height)

    px_ndc = - (px - image_width / 2.0) * 2.0 / min(image_width, image_height)
    py_ndc = - (py - image_height / 2.0) * 2.0 / min(image_width, image_height)

    focal_length = torch.tensor([fx_ndc, fy_ndc]).unsqueeze(0)
    principal_point = torch.tensor([px_ndc, py_ndc]).unsqueeze(0)
    return torch.cat([focal_length, principal_point], dim=1)
def find_all_generated_inputs(input_dir: Path) -> List[Path]:
    # rest of your code
    """
    Recursively visit an input directory, finding all generated directories.

    This function receives a root input directory, and collects all directories
    beginning with the "generated-*" prefix. It will recurse into subdirectories
    as long as they do not contain a "generated-*" directory.

    Args:
    ----
        input_dir: The root input directory.

    Returns:
    -------
        A list of all generated directories.
    """
    generated_dirs: list[Path] = []

    for input_path in input_dir.iterdir():
        if input_path.is_dir():
            if input_path.name.startswith("generated-") or "+0000" in input_path.name:
                generated_dirs.append(input_path)
            else:
                generated_dirs.extend(find_all_generated_inputs(input_path))

    generated_dirs.sort()
    return generated_dirs

def CreateSMPLX(gender, num_betas = 300):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # model_path = ROOT_DIR + "/smplx/smplx/models"
    model_path = "/home/ext_fares_podform3d_com/ProbabilisticNormals/modules/smplx/smplx/models"
    model_type = "smplx"
    gender = gender
    ext = "npz"
    num_betas = 300

    return smplx.create(model_path,
                         model_type=model_type,
                         gender=gender,
                         use_face_contour=False,
                         num_betas=num_betas,
                         ext=ext).to(device)

def extract_measurements(root_path: Path) -> torch.Tensor:

    betas = torch.load(root_path / "beta.pt")
    poses = torch.load(root_path / "pose.pt")
    D = torch.load(root_path / "displacement_vectors.pt")
    with open(root_path /'patient.json') as f:
        patient_data = json.load(f)
    gender = patient_data["sex"]
    model = CreateSMPLX(gender)
    output_model_nopose = model(betas=betas, D=D, return_verts=True)
    smplx_mesh_nopose = trimesh.Trimesh(vertices=output_model_nopose.vertices.detach().cpu().numpy().squeeze(),
                                  faces=model.faces, process=False, maintain_order=True)
    translation_nopose = torch.tensor([[0, - smplx_mesh_nopose.vertices.min(axis=0)[1], 0]],
                    device=device,
                    dtype=torch.float32)
    output_model_nopose = model(betas=betas, transl=translation_nopose, return_verts=True)
    smplx_mesh_nopose = trimesh.Trimesh(vertices=output_model_nopose.vertices.detach().cpu().numpy().squeeze(),
                                faces=model.faces, process=False, maintain_order=True)
    joints_nopose = output_model_nopose.joints
    
    right_leg_meas, left_leg_meas, torso_meas, lengths = extra_measurements_model(smplx_mesh_nopose, joints_nopose.squeeze().detach().cpu().numpy())

    width = np.concatenate((right_leg_meas[0], left_leg_meas[0], torso_meas[0]), axis=0)
    circ = np.concatenate((right_leg_meas[1], left_leg_meas[1], torso_meas[1]), axis=0)
    depth = np.concatenate((right_leg_meas[2], left_leg_meas[2], torso_meas[2]), axis=0)
    measurements = np.concatenate((width, circ, depth, lengths), axis=0)

    output_model_pose = model(betas=betas, body_pose=poses, D=D, return_verts=True)
    return torch.tensor(measurements, dtype=torch.float32), betas, poses, output_model_pose.joints, D, gender


def infer_samples(img_path, mask_path, args, model, intrins=None, device='cpu'):

    # target_path = "/home/ext_fares_podform3d_com/ProbabilisticNormals/data/data_chihab/probabilistic_normals"
    # target_path_im = os.path.join(target_path, "image")


    with torch.no_grad():
        mask = Image.open(mask_path)
        mask = mask.resize((720, 960), Image.BILINEAR)
        mask_np =np.array(mask)
        mask_np = np.where(mask_np >= 128, 255, mask_np)

        img = Image.open(img_path).convert('RGB')
        img = img.resize((720, 960), Image.BILINEAR)


        img = np.array(img).astype(np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        _, _, orig_H, orig_W = img.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        l, r, t, b = utils.pad_input(orig_H, orig_W)
        img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #img = normalize(img)

        intrins_path = "/home/ext_fares_podform3d_com/ProbabilisticNormals/data/v6.0-7000/intrins.txt"
        if os.path.exists(intrins_path):
           # NOTE: camera intrinsics should be given as a txt file
           # it should contain the values of fx, fy, cx, cy
           intrins = utils.get_intrins_from_txt(intrins_path, device=device).unsqueeze(0)
        else:
           # NOTE: if intrins is not given, we just assume that the principal point is at the center
           # and that the field-of-view is 60 degrees (feel free to modify this assumption)
           intrins = utils.get_intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)

        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t

        pred_norm = model(img, intrins=intrins)[-1]
        pred_norm = pred_norm[:, :, t:t+orig_H, l:l+orig_W]

        pred_norm_np = pred_norm.cpu().detach().numpy()[0,:,:,:].transpose(1, 2, 0) # (H, W, 3)
        pred_norm_np = ((pred_norm_np + 1.0) / 2.0 * 255.0).astype(np.uint8)

        # pred_norm_np = np.dstack((pred_norm_np, mask_np))
        # pred_norm_np = pred_norm_np * mask_np[:, :, np.newaxis]
        pred_norm_np = pred_norm_np[:, :, :3]
        
        mask_3d = np.expand_dims(mask, axis=-1)/255.0
        mask_3d= mask_3d.astype(np.uint8)

        return pred_norm_np * mask_3d


def test_samples(args, model, intrins=None, device='cpu'):

    target_path = "/home/ext_fares_podform3d_com/ProbabilisticNormals/data/data_chihab/probabilistic_normals"
    target_path_im = os.path.join(target_path, "image")





    img_path = "/home/ext_fares_podform3d_com/ProbabilisticNormals/data/app_outputs_april2024/April3_scans/2024-04-03 13_24_45 +0000-6CB0BF0A-7BB1-4824-A8D3-65F0E49348DF/data/frame_snapshots/00_snapshot/preprocessed/remove_bg_masked_rgb.png"

    mask_path =  "/home/ext_fares_podform3d_com/ProbabilisticNormals/data/app_outputs_april2024/April3_scans/2024-04-03 13_24_45 +0000-6CB0BF0A-7BB1-4824-A8D3-65F0E49348DF/data/frame_snapshots/00_snapshot/preprocessed/remove_bg_mask.png"
    with torch.no_grad():
        mask = Image.open(mask_path)
        mask = mask.resize((720, 960), Image.BILINEAR)
        mask_np =np.array(mask)
        mask_np = np.where(mask_np >= 128, 255, mask_np)

        img = Image.open(img_path).convert('RGB')
        img = img.resize((720, 960), Image.BILINEAR)


        img = np.array(img).astype(np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        _, _, orig_H, orig_W = img.shape

        # zero-pad the input image so that both the width and height are multiples of 32
        l, r, t, b = utils.pad_input(orig_H, orig_W)
        img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #img = normalize(img)

        intrins_path = "/home/ext_fares_podform3d_com/ProbabilisticNormals/data/v6.0-7000/intrins.txt"
        if os.path.exists(intrins_path):
           # NOTE: camera intrinsics should be given as a txt file
           # it should contain the values of fx, fy, cx, cy
           intrins = utils.get_intrins_from_txt(intrins_path, device=device).unsqueeze(0)
        else:
           # NOTE: if intrins is not given, we just assume that the principal point is at the center
           # and that the field-of-view is 60 degrees (feel free to modify this assumption)
           intrins = utils.get_intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)

        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t

        pred_norm = model(img, intrins=intrins)[-1]
        pred_norm = pred_norm[:, :, t:t+orig_H, l:l+orig_W]

        pred_norm_np = pred_norm.cpu().detach().numpy()[0,:,:,:].transpose(1, 2, 0) # (H, W, 3)
        pred_norm_np = ((pred_norm_np + 1.0) / 2.0 * 255.0).astype(np.uint8)

        pred_norm_np = np.dstack((pred_norm_np, mask_np))
        
        im = Image.fromarray(pred_norm_np)
        im.save("/home/ext_fares_podform3d_com/ProbabilisticNormals/data1/antonin2.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='dsine', type=str, help='model checkpoint')
    parser.add_argument('--mode', default='samples', type=str, help='{samples}')
    args = parser.parse_args()
    
    # define model
    device = torch.device('cuda')

    from models.dsine import DSINE
    model = DSINE().to(device)
    model.pixel_coords = model.pixel_coords.to(device)
    model = utils.load_checkpoint('./DSINE/checkpoints/%s.pt' % args.ckpt, model)
    model.eval()
    
    if args.mode == 'samples':
        test_samples(args, model, intrins=None, device=device)
