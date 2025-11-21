import os
import pickle
import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray
import shutil
import decord

# If you encounter an error that libc++.1.dylib is not found, resolve it with conda install libcxx.
# For headless rendering on a remote server, you may encounter an error like XOpenDisplay: cannot open display.
# In this case start a virtual display using Xvfb :0 -screen 0 640x480x24 & export DISPLAY=:0.0. 
# You may need to install xvfb first (apt install xvfb)

try:
    from aitviewer.configuration import CONFIG
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.billboard import Billboard
    from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
    from aitviewer.scene.camera import OpenCVCamera
    
    # 这个smplx模型的数据路径部署时需要复制到对应机器上
    CONFIG.smplx_models = os.path.join("/nvfile-heatstorage/human_guozz2/code/zhangyan/ml-comotion-main/src/comotion_demo/data")
    CONFIG.window_type = "pyqt6"
    # CONFIG.window_type = "pyglet"
    aitviewer_available = True

except ModuleNotFoundError:
    print(
        "WARNING: Skipped aitviewer import, ensure it is installed to run visualization."
    )
    aitviewer_available = False


def convert_image_to_tensor(image: NDArray[np.uint8]) -> torch.Tensor:
    """Convert an uint8 numpy array of shape HWC to a float tensor of shape CHW."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Expect input to be a numpy array.")
    if image.dtype != np.uint8:
        raise ValueError("Expect input to be np.uint8 typed.")
    return torch.from_numpy(image).permute(2, 0, 1)

def get_default_K(image: torch.Tensor) -> torch.Tensor:
    """Get a default approximate intrinsic matrix."""
    res = image.shape[-2:]
    max_res = max(res)
    K = torch.tensor([[2 * max_res, 0, 0.5 * res[1]], [0, 2 * max_res, 0.5 * res[0]]])
    return K

def query_range(tracks, i0, i1):
    to_use = (tracks["frame_idx"] >= i0) & (tracks["frame_idx"] < i1)
    filtered = {k: v[to_use] for k, v in tracks.items()}
    reshaped = {
        k: padded_reshaped(v, filtered["frame_idx"] - i0, filtered["id"], i1 - i0)
        for k, v in filtered.items()
    }
    return reshaped

def hsv2rgb(hsv):
    """Convert a tuple from HSV to RGB."""
    h, s, v = hsv
    vals = np.tile(v, [3] + [1] * v.ndim)
    vals[1:] *= 1 - s[None]

    h[h > (5 / 6)] -= 1
    diffs = np.tile(h, [3] + [1] * h.ndim) - (np.arange(3) / 3).reshape(
        3, *[1] * h.ndim
    )
    max_idx = np.abs(diffs).argmin(0)

    final_rgb = np.zeros_like(vals)

    for i in range(3):
        tmp_d = diffs[i] * (max_idx == i)
        dv = tmp_d * 6 * s * v
        vals[1] += np.maximum(0, dv)
        vals[2] += np.maximum(0, -dv)

        final_rgb += np.roll(vals, i, axis=0) * (max_idx == i)

    return final_rgb.transpose(*list(np.arange(h.ndim) + 1), 0)

def init_color_ref(n, seed=12345):
    """Sample N colors in HSV and convert them to RGB."""
    rand_state = np.random.RandomState(seed)
    rand_hsv = rand_state.rand(n, 3)
    rand_hsv[:, 1:] = 1 - rand_hsv[:, 1:] * 0.3
    color_ref = hsv2rgb(rand_hsv.T)

    return color_ref

def padded_reshaped(arr, frame_idxs, ids, num_frames):
    unique_ids, remapped_ids = ids.unique(return_inverse=True)
    max_tracks = len(unique_ids)
    dim_ref = [d for d in arr.shape]
    dim_ref = [num_frames, max_tracks] + dim_ref[1:]
    dst_arr = torch.zeros(*dim_ref, device=arr.device, dtype=arr.dtype)
    dst_arr[frame_idxs, remapped_ids] = arr

    return dst_arr

def prepare_scene(viewer, width, height, K, image_paths, fps=30):
    """Prepare the scene for AITViewer rendering."""
    viewer.reset()
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    extrinsics = np.eye(4)[:3]

    # Initialize camera
    cam = OpenCVCamera(K, extrinsics, cols=width, rows=height, viewer=viewer)
    viewer.scene.add(cam)
    viewer.scene.camera.position = [0, 0, -5]
    viewer.scene.camera.target = [0, 0, 10]
    viewer.auto_set_camera_target = False
    viewer.set_temp_camera(cam)
    viewer.playback_fps = fps

    # "billboard" display for video frames
    billboard = Billboard.from_camera_and_distance(
        cam, 100.0, cols=width, rows=height, textures=image_paths
    )
    viewer.scene.add(billboard)


def add_pose_to_scene(
    viewer,
    smpl_layer,
    betas,
    pose,
    trans,
    color=(0.6, 0.6, 0.6),
    alpha=1,
    color_ref=None,
):
    """Add estimated poses to the rendered scene."""
    if betas.ndim == 2:
        betas = betas[None]
        pose = pose[None]
        trans = trans[None]

    poses_root = pose[..., :3]
    poses_body = pose[..., 3:]
    max_people = pose.shape[1]

    if (betas != 0).any():
        for person_idx in range(max_people):
            if color_ref is None:
                person_color = color
            else:
                person_color = color_ref[person_idx % len(color_ref)] * 0.4 + 0.3
            person_color = [c_ for c_ in person_color] + [alpha]

            valid_vals = (betas[:, person_idx] != 0).any(-1)
            idx_range = valid_vals.nonzero()
            if len(idx_range) > 0:
                trans[~valid_vals][..., 2] = -10000
                viewer.scene.add(
                    SMPLSequence(
                        smpl_layer=smpl_layer,
                        betas=betas[:, person_idx],
                        poses_root=poses_root[:, person_idx],
                        poses_body=poses_body[:, person_idx],
                        trans=trans[:, person_idx],
                        color=person_color,
                    )
                )



def draw_3d_mesh(pkl_path, start_frame_id=0, max_frames=162, sample_step=1, first_frame_path=None, temp_save_path=None):
    # import pdb;pdb.set_trace()
    with open(pkl_path,'rb') as f:
        preds = pickle.load(f)
    if "bboxes" in preds.keys():
        bboxes = preds.pop("bboxes")
    else:
        bboxes = None
    first_frame = Image.open(first_frame_path)
    image_width, image_height = first_frame.size    

    pkl_name = pkl_path.split('/')[-1].split('.')[0]
    tmp_vis_dir = temp_save_path
    os.makedirs(tmp_vis_dir, exist_ok=True)

    image_paths = []
    #黑色背景纹理图片
    for frame_idx in range(max_frames//sample_step):
        image_paths.append(f"{tmp_vis_dir}/{frame_idx:06d}.jpg")
        # swap to black image.
        black_image = Image.new("RGB", (image_width, image_height), (0, 0, 0))
        black_image.save(image_paths[-1])
        frame_idx += 1

    # Initialize viewer
    viewer = HeadlessRenderer(size=(image_width, image_height))
    image = np.array(first_frame.convert("RGB"))
    image = convert_image_to_tensor(image)
    K = get_default_K(image)

    prepare_scene(viewer, image_width, image_height, K.cpu().numpy(), image_paths, fps=30)
    # Prepare SMPL poses
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")

    track_subset = query_range(preds, start_frame_id, start_frame_id+max_frames)
    print("track_subset",track_subset["pose"].shape)
    id_lookup = track_subset["id"].max(0)[0]
    color_ref_all = init_color_ref(2000)
    color_ref = color_ref_all[id_lookup % len(color_ref_all)]
    if len(id_lookup) == 1:
        color_ref = [color_ref]

    betas = track_subset["betas"][:,:1,:][::sample_step] # 默认取第一个id的pose
    pose = track_subset["pose"][:,:1,:][::sample_step]
    trans = track_subset["trans"][:,:1,:][::sample_step]
    # betas = torch.tensor([[-0.4664,  0.1617,  0.0370,  0.0064,  0.2755, -0.3656, -0.0448, -0.0082,
    #       0.0895, -0.0210]]) 
    # print("betas:",betas.shape)
    # betas = betas.expand(trans.shape[0], 1, 10) 
    print("betas:",betas[0], betas.shape, pose.shape, trans.shape)
    color=(0.6, 0.6, 0.6)
    alpha=1

    add_pose_to_scene(
        viewer, smpl_layer, betas, pose, trans, color, alpha, color_ref
    )

    # Save rendered scene
    save_path = f"{tmp_vis_dir}/{pkl_name}.mp4"
    viewer.save_video(
        video_dir=str(save_path),
        output_fps=30,
        ensure_no_overwrite=False,
    )
    
    # pose_data = decord.VideoReader(save_path)
    # frames = pose_data.get_batch(np.arange(0, max_frames//sample_step, dtype=int))
    # Remove temporary directory
    # shutil.rmtree(tmp_vis_dir)
    # viewer.scene.remove(*nodes)
    del viewer

    return 

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help="smpl path")
    parser.add_argument("--start_frame_id", type=int, default=0, required=False, help="格式为 0-7")
    parser.add_argument("--max_frames", type=int, default=81, required=False, help="格式为 0-7")
    parser.add_argument("--sample_step", type=int, default=2, required=False, help="格式为 0-7")
    parser.add_argument("--first_frame_path", type=str, required=True, help="first_frame_path")
    parser.add_argument("--temp_save_path", type=str, required=True, help="temp_save_path")

    args = parser.parse_args()
    
    draw_3d_mesh(args.pkl_path, args.start_frame_id, args.max_frames, args.sample_step, args.first_frame_path, args.temp_save_path)