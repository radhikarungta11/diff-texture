from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from .cameras import build_camera_from_pose
from .config import Config
from .io_utils import ensure_dirs, load_training_frames, save_json, save_preview, save_tensor_image, set_seed
from .losses import masked_l1, total_variation_loss
from .mesh_utils import build_textured_mesh, create_texture_logits, load_mesh_template, texture_from_logits
from .renderer import TextureOnlyRenderer
from .export_utils import export_textured_obj


class TextureOptimizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        ensure_dirs(cfg)
        set_seed(cfg.seed)

        self.mesh = load_mesh_template(cfg.mesh_path, cfg.device)
        self.frames = load_training_frames(cfg)
        self.texture_logits = create_texture_logits(cfg.texture_size, cfg.device)
        self.optimizer = Adam([self.texture_logits], lr=cfg.lr)
        self.renderer = TextureOnlyRenderer(
            image_size=cfg.image_size,
            background_rgb=cfg.background_rgb,
            device=cfg.device,
        )

        self.global_step = 0
        self.metrics = {
            'device': cfg.device,
            'num_frames': len(self.frames),
            'num_epochs': cfg.num_epochs,
            'texture_size': cfg.texture_size,
            'image_size': cfg.image_size,
            'history': [],
        }

    def _save_checkpoint(self) -> None:
        ckpt_path = self.cfg.checkpoint_dir / f'step_{self.global_step:06d}.pt'
        torch.save(
            {
                'step': self.global_step,
                'texture_logits': self.texture_logits.detach().cpu(),
                'optimizer': self.optimizer.state_dict(),
                'config': self.cfg.__dict__,
            },
            ckpt_path,
        )

    def _save_preview(self, frame_name: str, gt_rgb: torch.Tensor, gt_mask: torch.Tensor, pred_rgb: torch.Tensor) -> None:
        texture_map = texture_from_logits(self.texture_logits).detach()
        texture_vis = torch.nn.functional.interpolate(
            texture_map.permute(0, 3, 1, 2),
            size=(self.cfg.image_size, self.cfg.image_size),
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1)
        mask_vis = gt_mask.repeat(1, 1, 1, 3)
        preview_path = self.cfg.preview_dir / f'step_{self.global_step:06d}_{frame_name}.png'
        save_preview(preview_path, [gt_rgb, mask_vis, pred_rgb, texture_vis], labels=['gt', 'mask', 'pred', 'tex'])

    def _train_one_frame(self, frame: dict) -> dict[str, float]:
        pose = frame['pose']
        gt_rgb = frame['image']
        gt_mask = frame['mask']
        camera = build_camera_from_pose(pose, self.cfg.device)

        mesh = build_textured_mesh(self.mesh, self.texture_logits)
        pred_rgb, pred_alpha, _ = self.renderer(mesh, camera)

        texture_map = texture_from_logits(self.texture_logits)
        photo_loss = masked_l1(pred_rgb, gt_rgb, gt_mask)
        tv_loss = total_variation_loss(texture_map)
        loss = self.cfg.weight_photo * photo_loss + self.cfg.weight_tv * tv_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            'loss': float(loss.detach().item()),
            'photo_loss': float(photo_loss.detach().item()),
            'tv_loss': float(tv_loss.detach().item()),
            'alpha_mean': float(pred_alpha.detach().mean().item()),
        }

    def run(self) -> None:
        start = time.time()
        frame_indices = list(range(len(self.frames)))

        for epoch in range(1, self.cfg.num_epochs + 1):
            # shuffle views every epoch for better mixing
            perm = torch.randperm(len(frame_indices)).tolist()
            epoch_stats: list[dict[str, float]] = []

            pbar = tqdm(perm, desc=f'Epoch {epoch}/{self.cfg.num_epochs}', leave=False)
            for perm_idx in pbar:
                frame = self.frames[perm_idx]
                self.global_step += 1
                stats = self._train_one_frame(frame)
                epoch_stats.append(stats)
                pbar.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    photo=f"{stats['photo_loss']:.4f}",
                    tv=f"{stats['tv_loss']:.4f}",
                )

                if self.global_step % self.cfg.log_every == 0:
                    self.metrics['history'].append({
                        'step': self.global_step,
                        'epoch': epoch,
                        **stats,
                    })

                if self.global_step % self.cfg.preview_every == 0:
                    with torch.no_grad():
                        camera = build_camera_from_pose(frame['pose'], self.cfg.device)
                        mesh = build_textured_mesh(self.mesh, self.texture_logits)
                        pred_rgb, _, _ = self.renderer(mesh, camera)
                    self._save_preview(frame['name'], frame['image'], frame['mask'], pred_rgb)

                if self.global_step % self.cfg.checkpoint_every == 0:
                    self._save_checkpoint()

            mean_loss = sum(x['loss'] for x in epoch_stats) / max(1, len(epoch_stats))
            print(f'[epoch {epoch:03d}] mean_loss={mean_loss:.6f}')

        total_seconds = time.time() - start
        texture_map = texture_from_logits(self.texture_logits).detach()
        save_tensor_image(self.cfg.final_texture_path, texture_map)
        export_textured_obj(self.mesh, texture_map, self.cfg.final_obj_path)

        self.metrics['total_seconds'] = total_seconds
        self.metrics['final_texture_path'] = str(self.cfg.final_texture_path)
        self.metrics['final_obj_path'] = str(self.cfg.final_obj_path)
        save_json(self.cfg.final_metrics_path, self.metrics)

        print('\nDone.')
        print(f'Final texture: {self.cfg.final_texture_path}')
        print(f'Final OBJ:     {self.cfg.final_obj_path}')
        print(f'Metrics:       {self.cfg.final_metrics_path}')