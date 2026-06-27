from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torchv  
   

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'outputs'


def default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda:0'
    # PyTorch3D is usually not stable on Apple MPS yet.
    return 'cpu'


@dataclass
class Config:
    # Paths
    mesh_path: Path = DATA_DIR / 'mesh' / 'bottle_uv.obj'
    image_dir: Path = DATA_DIR / 'images'
    mask_dir: Path = DATA_DIR / 'masks'
    poses_path: Path = DATA_DIR / 'poses' / 'poses.json'

    preview_dir: Path = OUTPUT_DIR / 'previews'
    checkpoint_dir: Path = OUTPUT_DIR / 'checkpoints'
    final_dir: Path = OUTPUT_DIR / 'final'

    # Optimization setup
    image_size: int = 512
    texture_size: int = 1024
    num_epochs: int = 40
    lr: float = 0.05
    weight_photo: float = 1.0
    weight_tv: float = 0.002

    seed: int = 42
    device: str = default_device()

    # Logging / saving
    preview_every: int = 50
    checkpoint_every: int = 200
    log_every: int = 10

    # Misc
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def final_texture_path(self) -> Path:
        return self.final_dir / 'optimized_texture.png'

    @property
    def final_obj_path(self) -> Path:
        return self.final_dir / 'textured_bottle.obj'

    @property
    def final_metrics_path(self) -> Path:
        return self.final_dir / 'metrics.json'


CFG = Config()
