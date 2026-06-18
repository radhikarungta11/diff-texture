from __future__ import annotations

import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.utils import cameras_from_opencv_projection


def build_camera_from_pose(pose: dict, device: str) -> PerspectiveCameras:
    """
    Expected pose format per frame:
    {
      "file": "000000.png",
      "R": [[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]],
      "T": [tx, ty, tz],
      "fx": 1200.0,
      "fy": 1200.0,
      "cx": 512.0,
      "cy": 512.0,
      "width": 1024,
      "height": 1024
    }

    R and T should be OpenCV / COLMAP-style world-to-camera extrinsics.
    """
    R = torch.tensor(pose['R'], dtype=torch.float32, device=device)[None, ...]
    T = torch.tensor(pose['T'], dtype=torch.float32, device=device)[None, ...]

    fx = float(pose['fx'])
    fy = float(pose['fy'])
    cx = float(pose['cx'])
    cy = float(pose['cy'])
    width = int(pose['width'])
    height = int(pose['height'])

    camera_matrix = torch.tensor(
        [[
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]],
        dtype=torch.float32,
        device=device,
    )
    image_size = torch.tensor([[height, width]], dtype=torch.float32, device=device)

    cameras = cameras_from_opencv_projection(
        R=R,
        tvec=T,
        camera_matrix=camera_matrix,
        image_size=image_size,
    )
    return cameras