from __future__ import annotations

import torch
from torch import nn
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings


class TextureOnlyRenderer(nn.Module):
    """
    Rasterize the mesh and sample UV texture directly.
    This avoids lighting so the rendered RGB is purely texture color.
    """

    def __init__(self, image_size: int, background_rgb: tuple[float, float, float], device: str):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.background = torch.tensor(background_rgb, dtype=torch.float32, device=device).view(1, 1, 1, 3)

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            cull_backfaces=False,
            perspective_correct=True,
        )
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)

    def forward(self, meshes, cameras):
        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        texels = meshes.sample_textures(fragments)  # [N, H, W, K, 3]
        rgb = texels[..., 0, :]  # first visible face
        alpha = (fragments.pix_to_face[..., 0] >= 0).float()[..., None]
        rgb = rgb * alpha + self.background * (1.0 - alpha)
        return rgb, alpha, fragments