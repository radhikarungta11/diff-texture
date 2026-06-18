from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .config import Config


VALID_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: Config) -> None:
    cfg.preview_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.final_dir.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def list_image_files(folder: Path) -> list[Path]:
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    ])


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr)


def load_rgb(path: Path, image_size: int, device: str) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    tensor = pil_to_tensor(img)[None, ...].to(device)
    return tensor


def load_mask(path: Path, image_size: int, device: str) -> torch.Tensor:
    img = Image.open(path).convert('L')
    img = img.resize((image_size, image_size), resample=Image.NEAREST)
    tensor = pil_to_tensor(img)[None, ...].to(device)
    tensor = (tensor > 0.5).float()
    return tensor


def tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0.0, 1.0)
    if x.ndim == 4:
        x = x[0]
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 3)   
    arr = (x.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def save_tensor_image(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil_rgb(tensor).save(path)


def make_preview_grid(images: list[torch.Tensor], labels: list[str] | None = None) -> Image.Image:
    pil_images = [tensor_to_pil_rgb(img) for img in images]
    width, height = pil_images[0].size
    canvas = Image.new('RGB', (width * len(pil_images), height), color=(20, 20, 20))
    for i, img in enumerate(pil_images):
        canvas.paste(img, (i * width, 0))
    return canvas


def save_preview(path: Path, images: list[torch.Tensor], labels: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_preview_grid(images, labels=labels)
    grid.save(path)


def load_pose_records(poses_path: Path) -> list[dict[str, Any]]:
    data = load_json(poses_path)
    frames = data.get('frames', [])
    if not frames:
        raise ValueError(f'No frames found inside {poses_path}')
    required = {'file', 'R', 'T', 'fx', 'fy', 'cx', 'cy', 'width', 'height'}
    for idx, frame in enumerate(frames):
        missing = required - set(frame.keys())
        if missing:
            raise ValueError(f'Frame index {idx} is missing keys: {sorted(missing)}')
    return frames


def load_training_frames(cfg: Config) -> list[dict[str, Any]]:
    pose_records = load_pose_records(cfg.poses_path)
    frames: list[dict[str, Any]] = []
    for rec in pose_records:
        image_path = cfg.image_dir / rec['file']
        mask_path = cfg.mask_dir / rec['file']
        if not image_path.exists():
            raise FileNotFoundError(f'Image not found: {image_path}')
        if not mask_path.exists():
            raise FileNotFoundError(f'Mask not found: {mask_path}')

        frames.append({
            'name': image_path.stem,
            'image': load_rgb(image_path, cfg.image_size, cfg.device),
            'mask': load_mask(mask_path, cfg.image_size, cfg.device),
            'pose': rec,
        })
    return frames