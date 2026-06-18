from __future__ import annotations

from pathlib import Path

import torch
from pytorch3d.io import save_obj

from .mesh_utils import MeshTemplate


def export_textured_obj(mesh: MeshTemplate, texture_map: torch.Tensor, output_obj_path: Path) -> None:
    """
    PyTorch3D will also write the companion .mtl and texture image next to the .obj.
    """
    output_obj_path.parent.mkdir(parents=True, exist_ok=True)
    save_obj(
        f=output_obj_path,
        verts=mesh.verts.detach().cpu(),
        faces=mesh.faces.detach().cpu(),
        verts_uvs=mesh.verts_uvs.detach().cpu(),
        faces_uvs=mesh.faces_uvs.detach().cpu(),
        texture_map=texture_map[0].detach().cpu(),
    )