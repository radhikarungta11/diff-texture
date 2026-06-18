from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes


@dataclass
class MeshTemplate:
    verts: torch.Tensor
    faces: torch.Tensor
    verts_uvs: torch.Tensor
    faces_uvs: torch.Tensor


def load_mesh_template(mesh_path: Path, device: str) -> MeshTemplate:
    verts, faces, aux = load_obj(mesh_path, load_textures=False)

    if aux.verts_uvs is None or faces.textures_idx is None:
        raise ValueError(
            f'Mesh {mesh_path} does not contain UV data. '
            'Your OBJ must include vt entries and texture face indices.'
        )

    return MeshTemplate(
        verts=verts.to(device),
        faces=faces.verts_idx.to(device),
        verts_uvs=aux.verts_uvs.to(device),
        faces_uvs=faces.textures_idx.to(device),
    )


def create_texture_logits(texture_size: int, device: str, init_rgb: float = 0.5) -> nn.Parameter:
    init_rgb = max(1e-4, min(1.0 - 1e-4, init_rgb))
    init_logit = torch.logit(torch.tensor(init_rgb, dtype=torch.float32, device=device))
    logits = torch.full(
        (1, texture_size, texture_size, 3),
        fill_value=float(init_logit.item()),
        dtype=torch.float32,
        device=device,
    )
    return nn.Parameter(logits)


def texture_from_logits(texture_logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(texture_logits)


def build_textured_mesh(mesh: MeshTemplate, texture_logits: torch.Tensor) -> Meshes:
    texture_map = texture_from_logits(texture_logits)
    textures = TexturesUV(
        maps=texture_map,
        faces_uvs=mesh.faces_uvs[None, ...],
        verts_uvs=mesh.verts_uvs[None, ...],
    )
    textured_mesh = Meshes(
        verts=[mesh.verts],
        faces=[mesh.faces],
        textures=textures,
    )
    return textured_mesh