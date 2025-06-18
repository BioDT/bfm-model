"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""

import os
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Literal, Tuple

import hydra
import lightning as L
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import LightningModule, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig, OmegaConf
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset

from bfm_model.bfm.batch_utils import build_new_batch_with_prediction
from bfm_model.bfm.dataloader_monthly import (
    LargeClimateDataset,
    batch_to_device,
    custom_collate,
    debug_batch_devices,
    detach_batch,
    detach_graph_batch,
    detach_preds,
    inspect_batch_nans,
)
from bfm_model.bfm.decoder import BFMDecoder
from bfm_model.bfm.encoder import BFMEncoder
from bfm_model.bfm.utils import compute_next_timestamp, inspect_batch_shapes_namedtuple
from bfm_model.mvit.mvit_model import MViT
from bfm_model.swin_transformer.core.swim_core_v2 import Swin3DTransformer


def activation_ckpt_policy(module):
    return isinstance(module, (Swin3DTransformer, MViT))


class SequentialWindowDataset(Dataset):
    """
    Wrap an underlying single-sample dataset so __getitem__(i) returns a list
    [sample_i, sample_{i+1}, … sample_{i+steps}] where steps is user-defined.
    """

    def __init__(self, base_ds: Dataset, steps: int):
        assert steps >= 1, "steps must be ≥ 1"
        self.base = base_ds  # yields one Batch per index
        self.steps = steps

    def __len__(self):
        # last valid start idx is len(base) - steps - 1
        return len(self.base) - self.steps

    def __getitem__(self, idx):
        return [self.base[idx + k] for k in range(self.steps + 1)]  # list length = steps+1


class BFM_Forecastinglighting(LightningModule):
    """
    Biodiversity Foundation Model.

    This model combines encoder, backbone and decoder components to process climate and biodiversity-related variables.

    Args:
        surface_vars (tuple[str, ...]): Names of surface-level variables
        single_vars (tuple[str, ...]): Names of single-level variables
        atmos_vars (tuple[str, ...]): Names of atmospheric variables
        species_vars (tuple[str, ...]): Names of species-related variables
        species_distr_vars (tuple[str, ...]): Names of species distributions-related variables
        land_vars (tuple[str, ...]): Names of land-related variables
        agriculture_vars (tuple[str, ...]): Names of agriculture-related variables
        forest_vars (tuple[str, ...]): Names of forest-related variables
        atmos_levels (list[int]): Pressure levels for atmospheric variables
        species_num (int): Number of species distribution to account for
        H (int, optional): Height of output grid. Defaults to 32.
        W (int, optional): Width of output grid. Defaults to 64.
        num_latent_tokens (int, optional): Number of latent tokens. Defaults to 8.
        backbone_type (Literal["swin", "mvit"], optional): Type of backbone architecture. Defaults to "mvit".
        patch_size (int, optional): Size of spatial patches. Defaults to 4.
        embed_dim (int, optional): Embedding dimension. Defaults to 1024.
        num_heads (int, optional): Number of attention heads. Defaults to 16.
        head_dim (int, optional): Dimension of each attention head. Defaults to 64.
        depth (int, optional): Number of transformer layers. Defaults to 2.
        **kwargs: Additional arguments passed to encoder and decoder

    Attributes:
        encoder (BFMEncoder): Encoder component
        backbone (nn.Module): Backbone network (Swin or MViT)
        decoder (BFMDecoder): Decoder component
        backbone_type (str): Type of backbone being used
    """

    def __init__(
        self,
        surface_vars: tuple[str, ...],
        edaphic_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        climate_vars: tuple[str, ...],
        species_vars: tuple[str, ...],
        vegetation_vars: tuple[str, ...],
        land_vars: tuple[str, ...],
        agriculture_vars: tuple[str, ...],
        forest_vars: tuple[str, ...],
        redlist_vars: tuple[str, ...],
        misc_vars: tuple[str, ...],
        atmos_levels: list[int],
        species_num: int,
        H: int = 32,
        W: int = 64,
        num_latent_tokens: int = 8,
        backbone_type: Literal["swin", "mvit"] = "mvit",
        patch_size: int = 4,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 2,
        depth: int = 2,
        learning_rate: float = 5e-4,
        weight_decay: float = 5e-6,
        batch_size: int = 1,
        warmup_steps: int = 1000,
        total_steps: int = 20000,
        td_learning: bool = False,
        ground_truth_dataset=None,
        lead_time: int = 1,  # months
        refresh_interval=30,
        buffer_max_size=10,
        initial_buffer_size=5,
        peft_r: int = 8,
        lora_alpha: int = 8,
        d_initial: float = 0.1,
        peft_dropout: float = 0.0,
        peft_steps: int = 1,
        peft_mode: str = "single",
        use_lora: bool = False,
        use_vera: bool = False,
        rollout_steps: int = 1,
        swin_encoder_depths: Tuple[int, ...] = (2, 2, 2),
        swin_encoder_num_heads: Tuple[int, ...] = (8, 16, 32),
        swin_decoder_depths: Tuple[int, ...] = (2, 2, 2),
        swin_decoder_num_heads: Tuple[int, ...] = (32, 16, 8),
        swin_window_size: Tuple[int, ...] = (1, 4, 5),
        swin_mlp_ratio: float = 4.0,
        swin_qkv_bias: bool = True,
        swin_drop_rate: float = 0.0,
        swin_attn_drop_rate: float = 0.0,
        swin_drop_path_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.H = H
        self.W = W

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.td_learning = td_learning
        self.rollout_steps = rollout_steps
        self.lead_time = lead_time
        self.refresh_interval = refresh_interval
        # Store the ground truth dataset for matching.
        self.ground_truth_dataset = ground_truth_dataset
        # Build an iterator for refreshing.
        self.ground_truth_iter = iter(DataLoader(ground_truth_dataset, batch_size=1, shuffle=True, drop_last=False))
        # Initialize replay buffer and populate with ground truth samples.
        # self.replay_buffer = ReplayBuffer(max_size=buffer_max_size)
        # self.populate_replay_buffer(initial_buffer_size)
        # SAVE_DIR = pathlib.Path("rollout_exports")
        # SAVE_DIR.mkdir(exist_ok=True, parents=True)

        self.variable_weights = {
            "surface_variables": {
                "t2m": 0.1,
                "msl": 0.1,
                "slt": 0.1,
                "z": 0.1,
                "u10": 0.1,
                "v10": 0.1,
                "lsm": 0.1,
            },
            "edaphic_variables": {
                "swvl1": 0.1,
                "swvl2": 0.1,
                "stl1": 0.1,
                "stl2": 0.1,
            },
            "atmospheric_variables": {"z": 0.1, "t": 0.1, "u": 0.1, "v": 0.1, "q": 0.1},
            "climate_variables": {
                "smlt": 0.1,
                "tp": 0.1,
                "csfr": 0.1,
                "avg_sdswrf": 0.1,
                "avg_snswrf": 0.1,
                "avg_snlwrf": 0.1,
                "avg_tprate": 0.1,
                "avg_sdswrfcs": 0.1,
                "sd": 0.1,
                "t2m": 0.1,
                "d2m": 0.1,
            },
            "land_variables": {"Land": 0.1},
            "agriculture_variables": {"Agriculture": 0.1, "Arable": 0.1, "Cropland": 0.1},
            "forest_variables": {"Forest": 0.1},
            "redlist_variables": {"RLI": 0.1},
            "misc_variables": {"avg_slhtf": 0.1, "avg_pevr": 0.1},
            "species_variables": 100.0,
        }

        self.encoder = BFMEncoder(
            surface_vars=surface_vars,
            edaphic_vars=edaphic_vars,
            atmos_vars=atmos_vars,
            climate_vars=climate_vars,
            species_vars=species_vars,
            vegetation_vars=vegetation_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            redlist_vars=redlist_vars,
            misc_vars=misc_vars,
            atmos_levels=atmos_levels,
            species_num=species_num,
            H=H,
            W=W,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
            **kwargs,
        )

        patch_shape = (num_latent_tokens, H // self.encoder.patch_size, W // self.encoder.patch_size)

        if backbone_type == "swin":
            self.backbone = Swin3DTransformer(
                embed_dim=embed_dim,
                encoder_depths=swin_encoder_depths,
                encoder_num_heads=swin_encoder_num_heads,
                decoder_depths=swin_decoder_depths,
                decoder_num_heads=swin_decoder_num_heads,
                window_size=swin_window_size,
                mlp_ratio=swin_mlp_ratio,
                qkv_bias=swin_qkv_bias,
                drop_rate=swin_drop_rate,
                attn_drop_rate=swin_attn_drop_rate,
                drop_path_rate=swin_drop_path_rate,
                peft_r=peft_r,
                lora_alpha=lora_alpha,
                d_initial=d_initial,
                peft_dropout=peft_dropout,
                peft_steps=peft_steps,
                peft_mode=peft_mode,
                use_lora=use_lora,
                use_vera=use_vera,
            )
        elif backbone_type == "mvit":
            self.backbone = MViT(
                patch_shape=patch_shape,
                embed_dim=embed_dim,
                depth=4,
                num_heads=1,
                mlp_ratio=4.0,
                qkv_bias=True,
                path_drop_rate=0.1,
                attn_mode="conv",
                pool_first=False,
                rel_pos=False,
                zero_init_rel=True,
                res_pool=True,
                dim_mul_attn=False,
                dim_scales=[(i, 1.0) for i in range(4)],  # No dimension change
                head_scales=[(1, 2.0), (2, 2.0)],  # Keep head scaling for attention
                pool_kernel=[1, 1, 1],
                kv_stride=[1, 1, 1],
                q_stride=[(0, [1, 1, 1]), (1, [1, 1, 1]), (2, [1, 1, 1])],
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        self.backbone_type = backbone_type
        self.decoder = BFMDecoder(
            surface_vars=surface_vars,
            edaphic_vars=edaphic_vars,
            atmos_vars=atmos_vars,
            climate_vars=climate_vars,
            species_vars=species_vars,
            vegetation_vars=vegetation_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
            redlist_vars=redlist_vars,
            misc_vars=misc_vars,
            atmos_levels=atmos_levels,
            species_num=species_num,
            H=H,
            W=W,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
            **kwargs,
        )

        # Freeze pretrained parts.
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = False

        freeze_except(self)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{trainable/1e6:.2f} M / {total/1e6:.2f} M parameters will update")

    def forward(self, batch, lead_time: int = 1, batch_size: int = 1):
        """
        Forward pass of the model.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (timedelta): Time difference between input and target

        Returns:
            dict: Dictionary containing decoded outputs for each variable category

        """
        # print(f"BFM batch size: {batch_size}")
        # debug_batch_devices(batch)
        encoded = self.encoder(batch, lead_time, batch_size)
        # print("Encoded shape", encoded.shape)

        # calculate number of patches in 2D
        num_patches_h = self.H // self.encoder.patch_size
        num_patches_w = self.W // self.encoder.patch_size
        total_patches = num_patches_h * num_patches_w  # noqa

        # calculate depth to match the sequence length
        depth = encoded.shape[1] // (num_patches_h * num_patches_w)
        # print(f"BFM depth: {depth} | patch_size {self.encoder.patch_shape} | encoder shape {encoded.shape}")
        patch_shape = (
            depth,  # depth dimension matches sequence length / (H*W)
            num_patches_h,  # height in patches
            num_patches_w,  # width in patches
        )

        if self.backbone_type == "mvit":
            encoded = encoded.view(encoded.size(0), -1, self.encoder.embed_dim)
            print(f"Reshaped encoded for MViT: {encoded.shape}")

        backbone_output = self.backbone(encoded, lead_time=lead_time, rollout_step=0, patch_shape=patch_shape)
        # print("Backbone output", backbone_output.shape)
        # decode
        output = self.decoder(backbone_output, batch, lead_time)
        # print("Decoded output:", output)
        return output

    def rollout_forecast(self, initial_batch, steps: int = 1, batch_size: int = 1, mode: str = "finetune"):
        """
        If mode == 'finetune' -> pus-forward:
        steps 0 … K-2 run under no_grad (detached)
        step  K-1 keeps grad (memory = single step).

        If mode == 'eval' -> all steps under no_grad.
        """
        rollout_dict = {
            "batches": [],
            "timestamps": [],
            "lead_times": [],
        }
        device = next(self.parameters()).device
        curr = batch_to_device(initial_batch, device)
        for k in range(steps):
            keep_grad = mode == "finetune" and k == steps - 1
            with torch.set_grad_enabled(keep_grad):
                preds = self(curr, self.lead_time, batch_size=batch_size)
                # print(preds.keys())
            if keep_grad:
                next_batch = build_new_batch_with_prediction(curr, preds)
            else:
                # break graph but keep on GPU
                curr_det = detach_graph_batch(curr)
                preds_det = {g: {v: t.detach() for v, t in grp.items()} for g, grp in preds.items()}
                next_batch = build_new_batch_with_prediction(curr_det, preds_det)
            rollout_dict["batches"].append(next_batch)
            rollout_dict["timestamps"].append(next_batch.batch_metadata.timestamp)
            rollout_dict["lead_times"].append(next_batch.batch_metadata.lead_time)
            curr = next_batch
            # print(f"inspecting batch at step: {k}")
            # inspect_batch_shapes_namedtuple(curr)

        return rollout_dict

    def training_step(self, batch, batch_idx):
        """
        batch = [ Batch(t0,t1), Batch(t1,t2), … Batch(tK,tK+1) ]
                length = rollout_steps + 1
        """
        xs = batch
        init_batch = xs[0]
        target_batch = xs[self.rollout_steps]

        # push‑forward rollout
        roll = self.rollout_forecast(init_batch, steps=self.rollout_steps, batch_size=self.batch_size, mode="finetune")

        pred_last = roll["batches"][-1]  # Batch (tK,ŷK+1)
        loss = self.compute_loss(pred_last, target_batch)

        traj_loss = []
        # (optional) statistics on earlier steps without grads
        with torch.no_grad():
            for k in range(self.rollout_steps - 1):
                traj_loss.append(self.compute_loss(roll["batches"][k], xs[k + 1]))

        trajectory_loss = torch.stack(traj_loss).sum()
        self.log("train_loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("train_trajectory_loss", trajectory_loss, batch_size=self.batch_size, sync_dist=True)
        print(f"Train Loss: {loss} | {self.rollout_steps}-Step Trajectory Loss {trajectory_loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        xs = batch
        init_batch = xs[0]
        target_batch = xs[self.rollout_steps]
        # inspect_batch_shapes_namedtuple(init_batch)
        # inspect_batch_shapes_namedtuple(target_batch)
        traj_loss = []
        # push‑forward rollout
        roll = self.rollout_forecast(init_batch, steps=self.rollout_steps, batch_size=self.batch_size, mode="finetune")

        pred_last = roll["batches"][-1]  # Batch (tK,ŷK+1)
        inspect_batch_nans(roll["batches"][-1], tag="pred_last")
        inspect_batch_nans(target_batch, tag="gt_last")

        loss = self.compute_loss(pred_last, target_batch)
        # (optional) statistics on earlier steps without grads
        with torch.no_grad():
            for k in range(self.rollout_steps - 1):
                traj_loss.append(self.compute_loss(roll["batches"][k], xs[k + 1]))

        trajectory_loss = torch.stack(traj_loss).sum()
        self.log("val_loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("val_trajectory_loss", trajectory_loss, batch_size=self.batch_size, sync_dist=True)
        print(f"Val Loss: {loss} | {self.rollout_steps}-Step Trajectory Loss {trajectory_loss}")

        return loss

    def test_step(self, batch, batch_idx):
        xs = batch
        init_batch = xs[0]
        target_batch = xs[self.rollout_steps]
        traj_loss = []
        # push‑forward rollout
        roll = self.rollout_forecast(init_batch, steps=self.rollout_steps, batch_size=self.batch_size, mode="finetune")

        pred_last = roll["batches"][-1]  # Batch (tK,ŷK+1)
        loss = self.compute_loss(pred_last, target_batch)

        # (optional) statistics on earlier steps without grads
        with torch.no_grad():
            for k in range(self.rollout_steps - 1):
                traj_loss.append(self.compute_loss(roll["batches"][k], xs[k + 1]))

        trajectory_loss = torch.stack(traj_loss).sum()
        self.log("test_loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("test_trajectory_loss", trajectory_loss, batch_size=self.batch_size, sync_dist=True)
        print(f"Test Loss: {loss} | {self.rollout_steps}-Step Trajectory Loss {trajectory_loss}")
        return loss

    def predict_step(self, batch, batch_idx):
        # batch lives on GPU in fp16/bf16
        init = batch[0]
        init = batch_to_device(init, next(self.parameters()).device)

        rollout = self.rollout_forecast(init, steps=self.rollout_steps, batch_size=self.batch_size, mode="test")[
            "batches"
        ]  # list of Batches on GPU

        records = []
        for k, (pred, gt) in enumerate(zip(rollout, batch[1:]), start=1):
            # detach + clone + move to CPU
            pred_cpu = detach_batch(pred)
            gt_cpu = detach_batch(gt)
            records.append(
                {
                    "idx": batch_idx,
                    "step": k,
                    "pred": pred_cpu,
                    "gt": gt_cpu,
                }
            )
        return records

    # Will show all the layers we freeze!
    # def on_after_backward(self):
    #     """
    #     Checker for not learnable parameters -> Should output nothing!
    #     """
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def compute_loss(self, output, batch):
        # batch = target:      t1, t2
        # output = prediction: t1, t2
        total_loss = 0.0
        count = 0

        groups = [
            "surface_variables",
            "edaphic_variables",
            "atmospheric_variables",
            "climate_variables",
            "species_variables",
            "vegetation_variables",
            "land_variables",
            "agriculture_variables",
            "forest_variables",
            "redlist_variables",
            "misc_variables",
        ]
        for group_name in groups:

            pred_dict = getattr(output, group_name)
            true_dict = getattr(batch, group_name)
            # print(f"pred_dict {pred_dict} \n  TRUE DICT {true_dict}")
            group_loss = 0.0
            var_count = 0
            # inspect_batch_shapes_namedtuple(pred_dict)
            for var_name, pred_tensor in pred_dict.items():
                # If var_name not in the ground truth dict, skip
                if var_name not in true_dict:
                    print(f"{var_name} not in true_dict")
                    continue
                gt_tensor = true_dict[var_name]

                if self.td_learning:
                    time0 = gt_tensor[:, 0]
                    time1 = gt_tensor[:, 1]

                    true_diff = time1 - time0
                    pred_diff = pred_tensor[:, 1] - time0
                    # print("Pred and true dif shapes", pred_diff.shape, true_diff.shape)

                    loss_var = torch.mean(torch.abs(pred_diff - true_diff))
                else:
                    # print(f"prediction tensor shape: {pred_tensor.shape}")
                    time1 = gt_tensor[:, 1]
                    pred = pred_tensor[:, 1]

                    # mask out NaN/Inf before loss
                    mask = torch.isfinite(time1) & torch.isfinite(pred)
                    if mask.sum() == 0:
                        continue
                    p_clean = pred[mask]
                    t_clean = time1[mask]

                    loss_var = torch.mean(torch.abs(p_clean - t_clean))

                # Determine the weight for this variable.
                group_weights = self.variable_weights.get(group_name, {})
                if isinstance(group_weights, dict):
                    w = group_weights.get(var_name, 1.0)
                else:
                    w = group_weights
                # Log each variable's raw loss
                self.log(
                    f"{var_name} raw loss", loss_var, batch_size=time1.size(0), sync_dist=True
                )  # to accumulate the metric across devices.
                group_loss += w * loss_var
                var_count += 1

            if var_count > 0:
                group_loss /= var_count  # average within group
                total_loss += group_loss
                count += 1

        if count > 0:
            total_loss /= count  # average across groups

        # print(f"Single step Loss: {total_loss}")
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=self.learning_rate / 10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)

        return [optimizer]


def freeze_except(model):
    """
    - sets requires_grad=True  only for tensors whose name contains PEFT variant
    e.g. 'lora_' or 'vera_'
    - everything else is frozen
    - returns list of trainable parameter names for sanity-check
    """
    trainable = []
    for name, param in model.named_parameters():
        if "peft_" in name:
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    print(f"PEFT trainable params = {len(trainable)} layers")
    return trainable


class RolloutSaveCallback(Callback):
    SAVE_DIR = Path("rollout_exports")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        """
        `outputs` is a list of lists of dicts as returned by predict_step.
        """
        flat = [rec for batch_list in outputs for rec in batch_list]
        by_window = {}
        for rec in flat:
            idx = rec["idx"]
            by_window.setdefault(idx, []).append(rec)

        for idx, recs in by_window.items():
            recs = sorted(recs, key=lambda r: r["step"])
            path = self.SAVE_DIR / f"window_{idx:05d}.pt"
            torch.save(recs, path)
            pl_module.print(f"Saved {path} ({len(recs)} steps)")


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    """
    Test/inference script using a PyTorch Lightning module.

    Args:
        checkpoint_path (str): Path to the trained checkpoint (.ckpt).
        data_dir (str): Directory containing test data.
        batch_size (int): Batch size for test loader.
        num_workers (int): Number of workers for DataLoader.
        gpus (int): Number of GPUs to use (if 0, run on CPU).
        precision (int): Float precision (16 for half, 32 for single, etc.).
        accelerator (str): "gpu", "cpu", "tpu", etc.

    Returns:
        test_results (dict or list): Test metrics returned by trainer.test().
    """
    MODE = cfg.finetune.mode
    # Setup config
    print(OmegaConf.to_yaml(cfg))
    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory: {output_dir}")
    dataset = LargeClimateDataset(
        data_dir=cfg.data.data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        mode="finetune",
        atmos_levels=cfg.data.atmos_levels,
    )
    test_dataset = LargeClimateDataset(
        data_dir=cfg.data.test_data_path,
        scaling_settings=cfg.data.scaling,
        num_species=cfg.data.species_number,
        mode="finetune",
        atmos_levels=cfg.data.atmos_levels,
    )
    seq_dataset = SequentialWindowDataset(dataset, cfg.finetune.rollout_steps)
    seq_test_dataset = SequentialWindowDataset(test_dataset, cfg.finetune.rollout_steps)

    val_dataloader = DataLoader(
        seq_test_dataset,
        batch_size=cfg.finetune.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=False,
        shuffle=False,
    )

    train_dataloader = DataLoader(
        seq_dataset,
        shuffle=False,  # We need to keep the dates
        batch_size=cfg.finetune.batch_size,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Setting up Daloaders with length train: {len(train_dataloader)} and test: {len(val_dataloader)}")
    current_time = datetime.now()
    rank = int(os.environ.get("RANK", "0"))
    print(f"Will be using rank {rank} for logging")
    # Single logger approach with rank-specific paths
    mlf_logger = None
    # if "RANK" not in os.environ or os.environ["RANK"] == "0":
    if rank == 0 or rank == "0":
        # Use rank in experiment name to avoid conflicts
        mlf_logger = MLFlowLogger(
            experiment_name=f"BFM_{MODE}_finetune_logs_r{rank}",
            run_name=f"BFM_{MODE}_finetune_{current_time}",
            save_dir=f"{output_dir}/logs/rank{rank}",
        )

    checkpoint_path = cfg.finetune.checkpoint_path
    # Load Model from Checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")

    swin_params = {}
    if cfg.model.backbone == "swin":
        selected_swin_config = cfg.model_swin_backbone[cfg.model.swin_backbone_size]
        swin_params = {
            "swin_encoder_depths": tuple(selected_swin_config.encoder_depths),
            "swin_encoder_num_heads": tuple(selected_swin_config.encoder_num_heads),
            "swin_decoder_depths": tuple(selected_swin_config.decoder_depths),
            "swin_decoder_num_heads": tuple(selected_swin_config.decoder_num_heads),
            "swin_window_size": tuple(selected_swin_config.window_size),
            "swin_mlp_ratio": selected_swin_config.mlp_ratio,
            "swin_qkv_bias": selected_swin_config.qkv_bias,
            "swin_drop_rate": selected_swin_config.drop_rate,
            "swin_attn_drop_rate": selected_swin_config.attn_drop_rate,
            "swin_drop_path_rate": selected_swin_config.drop_path_rate,
        }

    BFM = BFM_Forecastinglighting.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        surface_vars=(cfg.model.surface_vars),
        edaphic_vars=(cfg.model.edaphic_vars),
        atmos_vars=(cfg.model.atmos_vars),
        climate_vars=(cfg.model.climate_vars),
        species_vars=(cfg.model.species_vars),
        vegetation_vars=(cfg.model.vegetation_vars),
        land_vars=(cfg.model.land_vars),
        agriculture_vars=(cfg.model.agriculture_vars),
        forest_vars=(cfg.model.forest_vars),
        redlist_vars=(cfg.model.redlist_vars),
        misc_vars=(cfg.model.misc_vars),
        atmos_levels=cfg.data.atmos_levels,
        species_num=cfg.data.species_number,
        H=cfg.model.H,
        W=cfg.model.W,
        num_latent_tokens=cfg.model.num_latent_tokens,
        backbone_type=cfg.model.backbone,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        depth=cfg.model.depth,
        learning_rate=cfg.finetune.lr,
        weight_decay=cfg.finetune.wd,
        batch_size=cfg.finetune.batch_size,
        td_learning=cfg.finetune.td_learning,
        ground_truth_dataset=test_dataset,
        strict=False,  # False if loading from a pre-trained with PEFT checkpoint
        peft_r=cfg.finetune.rank,
        lora_alpha=cfg.finetune.lora_alpha,
        d_initial=cfg.finetune.d_initial,
        peft_dropout=cfg.finetune.peft_dropout,
        peft_steps=cfg.finetune.rollout_steps,
        peft_mode=cfg.finetune.peft_mode,
        use_lora=cfg.finetune.use_lora,
        use_vera=cfg.finetune.use_vera,
        rollout_steps=cfg.finetune.rollout_steps,
        # lora_steps=cfg.finetune.rollout_steps, # 1 month
        # lora_mode=cfg.finetune.lora_mode, # every step + layers #single
        **swin_params,
    )

    apply_activation_checkpointing(BFM, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=activation_ckpt_policy)

    model_summary = ModelSummary(BFM, max_depth=2)
    print(model_summary)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        save_top_k=1,
        monitor="val_loss",  # `log('val_loss', value)` in the `LightningModule`
        mode="min",
        every_n_train_steps=cfg.finetune.checkpoint_every,
        filename="{epoch:02d}-{train_loss}",
        save_last=True,
    )

    print(f"Will be saving checkpoints at: {output_dir}/checkpoints")

    if cfg.training.strategy == "fsdp":
        distr_strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=1e6),
            # activation_checkpointing_policy=activation_ckpt_policy,
        )
        print(f"Using {cfg.training.strategy} strategy: {distr_strategy}")
    elif cfg.training.strategy == "ddp":
        distr_strategy = DDPStrategy()
        print(f"Using {cfg.training.strategy} strategy: {distr_strategy}")
    else:
        distr_strategy = "auto"

    # TODO: If this is not set, it's complaining about unused parameters
    distr_strategy = "ddp_find_unused_parameters_true"

    trainer = L.Trainer(
        max_epochs=cfg.finetune.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        strategy=distr_strategy,
        num_nodes=cfg.training.num_nodes,
        log_every_n_steps=cfg.training.log_steps,
        logger=mlf_logger,  # Only the rank 0 process will have a logger
        # limit_train_batches=10,      # Process 10 batches per epoch.
        # limit_val_batches=10,
        # limit_test_batches=2,
        # limit_predict_batches=2,
        val_check_interval=cfg.finetune.eval_every,  # Run validation every n training batches.
        check_val_every_n_epoch=None,
        # limit_train_batches=1, # For debugging to see what happens at the end of epoch
        # check_val_every_n_epoch=None,  # Do eval every n epochs
        # val_check_interval=3, # Does not work in Distributed settings | Do eval every 10 training steps => 10 steps x 8 batch_size = Every 80 Batches
        callbacks=[checkpoint_callback],
        # callbacks=[RolloutSaveCallback()],
        # plugins=[MyClusterEnvironment()],
    )

    if cfg.finetune.prediction:
        print(f"Will be doing {cfg.finetune.rollout_steps} - steps prediction and storing the results")
        all_outputs = trainer.predict(BFM, dataloaders=val_dataloader)

        # Flatten the nested lists:
        flat = [rec for batch_list in all_outputs for rec in batch_list]

        # Group by window index and save:
        SAVE_DIR = Path("rollout_exports")
        SAVE_DIR.mkdir(exist_ok=True, parents=True)

        by_window = {}
        for rec in flat:
            idx = rec["idx"]
            by_window.setdefault(idx, []).append(rec)

        for idx, recs in by_window.items():
            recs = sorted(recs, key=lambda r: r["step"])
            path = SAVE_DIR / f"window_{idx:05d}.pt"
            # each rec already has pred/gt = detached CPU Batches
            torch.save(recs, path)
            print(f"Saved {path} ({len(recs)} steps)")
    else:
        print(f"Starting {MODE} Finetune training from scratch for a horizon of {cfg.finetune.rollout_steps} ")
        trainer.fit(BFM, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    if dist.is_initialized():
        dist.barrier()

    selected_ckpt = checkpoint_callback.best_model_path or cfg.finetune.checkpoint_path
    if not os.path.exists(selected_ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {selected_ckpt}")

    if trainer.is_global_zero:
        print(f"[Rank 0] Using checkpoint: {selected_ckpt}")

    # broadcast checkpoint path
    if dist.is_initialized():
        ckpt_list = [selected_ckpt]
        dist.broadcast_object_list(ckpt_list, src=0)
        selected_ckpt = ckpt_list[0]


if __name__ == "__main__":
    main()
