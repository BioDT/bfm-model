"""
Copyright (C) 2025 TNO, The Netherlands. All rights reserved.
"""
import copy
import os
from typing import Literal

from datetime import datetime, timedelta
import random
from collections import deque, namedtuple
import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything, LightningModule
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from bfm_model.bfm.batch_utils import save_batch
from bfm_model.bfm.dataloder import LargeClimateDataset, custom_collate
from bfm_model.bfm.train_lighting import BFM_lighting
from bfm_model.bfm.utils import compute_next_timestamp, inspect_batch_shapes_namedtuple
from bfm_model.bfm.rollouts import build_new_batch_with_prediction

from bfm_model.bfm.decoder import BFMDecoder
from bfm_model.bfm.encoder import BFMEncoder
from bfm_model.mvit.mvit_model import MViT
from bfm_model.swin_transformer.core.swim_core_v2 import Swin3DTransformer



class SequentialWindowDataset(Dataset):
    """
    Wrap an underlying single‑sample dataset so __getitem__(i) returns a list
    [sample_i, sample_{i+1}, … sample_{i+steps}] where steps is user‑defined.
    """
    def __init__(self, base_ds: Dataset, steps: int):
        assert steps >= 1, "`steps` must be ≥ 1"
        self.base = base_ds          # yields one Batch per index
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
        single_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        species_vars: tuple[str, ...],
        species_distr_vars: tuple[str, ...],
        land_vars: tuple[str, ...],
        agriculture_vars: tuple[str, ...],
        forest_vars: tuple[str, ...],
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
        td_learning: bool = True,
        use_lora: bool = True,
        ground_truth_dataset=None, 
        rollout_steps: int = 2,
        lead_time: int = 6, # hours
        refresh_interval=30,
        buffer_max_size=10, 
        initial_buffer_size=5,
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
        self.lead_time = timedelta(hours = lead_time)
        self.refresh_interval = refresh_interval
        # Store the ground truth dataset for matching.
        self.ground_truth_dataset = ground_truth_dataset
        # Build an iterator for refreshing.
        self.ground_truth_iter = iter(DataLoader(ground_truth_dataset, batch_size=1, shuffle=True, drop_last=False))
        # Initialize replay buffer and populate with ground truth samples.
        # self.replay_buffer = ReplayBuffer(max_size=buffer_max_size)
        # self.populate_replay_buffer(initial_buffer_size)


        self.variable_weights = {
            "surface_variables": {
                "t2m": 1.7,
                "msl": 1.5,
                # ... add more if surface has more
            },
            "single_variables": {"lsm": 0.32},
            "atmospheric_variables": {"z": 0.46, "t": 1.2},
            "species_extinction_variables": {"ExtinctionValue": 1.43},
            "land_variables": {"Land": 0.2, "NDVI": 1.48},
            "agriculture_variables": {
                "AgricultureLand": 0.4,
                "AgricultureIrrLand": 0.92,
                "ArableLand": 0.38,
                "Cropland": 0.51,
            },
            "forest_variables": {"Forest": 0.38},
            "species_variables": {"Distribution": 2.0},
        }

        self.encoder = BFMEncoder(
            surface_vars=surface_vars,
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            species_vars=species_vars,
            species_distr_vars=species_distr_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
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
                encoder_depths=(2, 2),
                encoder_num_heads=(8, 16),
                decoder_depths=(2, 2),
                decoder_num_heads=(32, 16),
                window_size=(1, 1, 2),
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                use_lora=use_lora,
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
            single_vars=single_vars,
            atmos_vars=atmos_vars,
            species_vars=species_vars,
            species_distr_vars=species_distr_vars,
            land_vars=land_vars,
            agriculture_vars=agriculture_vars,
            forest_vars=forest_vars,
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

    def forward(self, batch, lead_time=timedelta(hours=6), batch_size: int = 1):
        """
        Forward pass of the model.

        Args:
            batch: Batch object containing input variables and metadata
            lead_time (timedelta): Time difference between input and target

        Returns:
            dict: Dictionary containing decoded outputs for each variable category

        """
        # print(f"BFM batch size: {batch_size}")
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

    # def populate_replay_buffer(self, num_samples):
    #     # Populate buffer with num_samples from ground truth dataset.
    #     dl = DataLoader(self.ground_truth_dataset, batch_size=1, shuffle=False, drop_last=False)
    #     for i, sample in enumerate(dl):
    #         if i >= num_samples:
    #             break
    #         self.replay_buffer.push(BufferItem(batch=sample, is_rollout=False))
        
    def rollout_forecast(self, initial_batch, steps=1, batch_size=1):

        # Container for results
        rollout_dict = {
            "predictions": [],
            "batches": [],
            "timestamps": [],
            "lead_times": [],
        }
        # print("Initial batch",initial_batch)
        current_batch = copy.deepcopy(initial_batch)

        # For each step in the rollout
        for step_idx in range(steps):
            # run predict
            self.eval()
            with torch.no_grad():
                preds = self(current_batch, self.lead_time, batch_size=batch_size)

            # preds = run_predict_on_batch(current_batch)  # shape depends on your model, e.g. [B, C, H, W]

            # store
            rollout_dict["predictions"].append(preds)
            rollout_dict["batches"].append(copy.deepcopy(current_batch))

            # handle times
            # Suppose your "Batch" has metadata => lead_time, timestamps...
            # Store the new predicted time.
            step_timestamp = current_batch.batch_metadata.timestamp[-1]
            rollout_dict["timestamps"].append(step_timestamp)
            rollout_dict["lead_times"].append(current_batch.batch_metadata.lead_time)

            # Build a new batch that has (last old time) + (predicted new time).
            new_batch = build_new_batch_with_prediction(current_batch, preds)

            # This new_batch becomes the "current_batch" for the next iteration
            current_batch = new_batch

        return rollout_dict


    def validation_step(self, batch, batch_idx):
        lead_time = timedelta(hours=6)  # fixed lead time for pre-training
        output = self(batch, lead_time, batch_size=self.batch_size)
        print("Validation time!")
        loss = self.compute_loss(output, batch)
        self.log("val_loss", loss, batch_size=self.batch_size)  # on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        return loss

    def training_step(self, batch, batch_idx):

        # buffer_item = self.replay_buffer.sample(batch_size=1)[0]
        initial_batch = batch[0] # Expected shape [B,2,...] , select the first only
        print(f"DEBUG: training_step - sampled batch timestamp: {initial_batch.batch_metadata.timestamp}")

        rollout_result = self.rollout_forecast(initial_batch, steps=self.rollout_steps, batch_size=1)
        print("Rollout timestamps",rollout_result["timestamps"])
        total_loss = 0.0
        valid_steps = 0
        rb_count = 0
        for rb in rollout_result["batches"]:
            print(f"Batch {rb_count}")
            # Use the second timestamp as the target.
            ts = rb.batch_metadata.timestamp
            target_ts = ts[rb_count] if isinstance(ts, list) and len(ts) > 1 else ts
            print(f"DEBUG: training_step - evaluating rollout batch with target timestamp: {target_ts}")
            rb_count+=1
            output = self(rb, self.lead_time, batch_size=self.batch_size)
            loss = self.compute_loss(output, rb)
            print(f"DEBUG: training_step - evaluating rollout batch with target timestamp: {target_ts}")

            total_loss += loss
            valid_steps += 1

        if valid_steps > 0:
            total_loss = total_loss / valid_steps
        else:
            total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Update replay buffer: add the last rollout batch (as a rollout sample).
        # self.replay_buffer.push(BufferItem(batch=rollout_result["batches"][-1], is_rollout=True))
        # print(f"DEBUG: training_step - no matching ground truth found for timestamp {target_ts}")

        # Periodically refresh the buffer with a fresh ground truth sample.
        # if self.global_step % self.refresh_interval == 0:
        #     dl = DataLoader(self.ground_truth_dataset, batch_size=1, shuffle=True, drop_last=True)
        #     fresh_sample = next(iter(dl))
        #     self.replay_buffer.push(BufferItem(batch=fresh_sample, is_rollout=False))
        #     print(f"DEBUG: training_step - refreshed replay buffer with new ground truth sample: {fresh_sample.batch_metadata.timestamp}")

        self.log("train_loss", total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        lead_time = timedelta(hours=6)  # fixed lead time for pre-training
        output = self(batch, lead_time, batch_size=self.batch_size)
        print("Test time")
        loss = self.compute_loss(output, batch)
        self.log("test_loss", loss, batch_size=self.batch_size)
        return loss

    def predict_step(self, batch, batch_idx):
        lead_time = timedelta(hours=6)  # fixed lead time for pre-training
        output = self(batch, lead_time, batch_size=self.batch_size)
        return output
    
    def compute_loss(self, output, batch):
        """
        Computes an average trajectory loss over multiple rollout timesteps.
        
        Assumptions:
        - For each modality (group) and variable, the prediction tensor (from output)
            has shape [B, T_pred, ...]. 
        - The ground truth in the batch has shape [B, T_gt, ...] with T_gt = T_pred + 1,
            where the 0th time index is the initial state and indices 1...T_gt-1 correspond 
            to the rollout target states.
        
        Two loss options are provided:
        - If self.td_learning is True, then for each rollout step t (from 1 to T_gt-1)
            we compute:
            loss_t = mean(|(pred[t-1] - gt[t-1]) - (gt[t] - gt[t-1])|).
        - Otherwise, we compute the L₁ error directly by comparing
            pred[t-1] with gt[t], averaged over the trajectory.
        
        The loss is averaged over all variables in a group and finally across groups.
        """
        total_loss = 0.0
        count = 0

        groups = [
            "surface_variables",
            "single_variables",
            "atmospheric_variables",
            "species_extinction_variables",
            "land_variables",
            "agriculture_variables",
            "forest_variables",
            "species_variables",
        ]

        for group_name in groups:
            # Skip groups not present in either output or ground truth.
            if group_name not in output or group_name not in batch._asdict():
                continue

            pred_dict = output[group_name]    # Expect a dict: var_name -> predicted tensor [B, T_pred, ...]
            true_dict = getattr(batch, group_name)  # Expect a dict: var_name -> ground truth tensor [B, T_gt, ...]

            group_loss = 0.0
            var_count = 0

            for var_name, pred_tensor in pred_dict.items():
                if var_name not in true_dict:
                    print(f"{var_name} not in true_dict")
                    continue

                gt_tensor = true_dict[var_name]
                # Assume: gt_tensor.shape = [B, T_gt, ...] and pred_tensor.shape = [B, T_pred, ...]
                # Here T_pred should equal T_gt - 1.
                B = gt_tensor.size(0)
                T_gt = gt_tensor.size(1)
                T_pred = pred_tensor.size(1)  # expected T_pred = T_gt - 1

                if self.td_learning:
                    # Use temporal difference loss.
                    loss_var = 0.0
                    # Loop from t=1 to T_gt-1.
                    for t in range(1, T_gt):
                        # The ground truth increment.
                        true_diff = gt_tensor[:, t] - gt_tensor[:, t - 1]
                        # The predicted increment: note that predictions index 0 corresponds to target t=1.
                        pred_diff = pred_tensor[:, t - 1] - gt_tensor[:, t - 1]
                        loss_var += torch.mean(torch.abs(pred_diff - true_diff))
                    # Average the loss across trajectory steps.
                    loss_var = loss_var / (T_gt - 1)
                else:
                    # Direct forecast loss: compare the prediction at each rollout step with ground truth.
                    # Here we compare pred_tensor[:, t] with gt_tensor[:, t+1] for t in 0...T_pred-1.
                    loss_var = torch.mean(torch.abs(pred_tensor - gt_tensor[:, 1:]))
                
                # Weight the loss per variable.
                group_weights = self.variable_weights.get(group_name, {})
                if isinstance(group_weights, dict):
                    w = group_weights.get(var_name, 1.0)
                else:
                    w = group_weights

                # Log the raw loss for this variable.
                self.log(f"{var_name} raw trajectory loss", loss_var, batch_size=B)
                group_loss += w * loss_var
                var_count += 1

            if var_count > 0:
                group_loss /= var_count  # average within group
                total_loss += group_loss
                count += 1

        if count > 0:
            total_loss /= count  # average across groups

        print(f"Trajectory Loss: {total_loss}")
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12000, eta_min=self.learning_rate / 10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)

        return [optimizer], [scheduler]

def fetch_ground_truth(next_timestamp, dataset):
    """
    Given a timestamp, fetch the corresponding ground truth Batch from the dataset.
    
    In our setting the dataset supplies ground truth samples that are pairs: 
    each sample is a Batch with [t-1, t].  To get a pair [t, t+1],
    we search the dataset (or use an index) such that the first timestep
    matches 'next_timestamp'.
    
    Here we simulate it by assuming our dataset is indexed sequentially.
    
    Args:
        next_timestamp: the desired timestamp for the first timestep.
        dataset: a Dataset object that returns Batch objects.
        
    Returns:
        batch_gt: a Batch whose first timestep corresponds to next_timestamp,
                  and the second timestep is the ground truth for t+1.
                  Returns None if no match is found.
    """
    # For simplicity, we iterate over the dataset and find a match.
    # TODO optimized (e.g. by maintaining a dict index).
    for sample in dataset:
        print("Dataset sample timestamp", sample.batch_metadata.timestamp)
        print("Timestamp looking for", next_timestamp)
        if sample.batch_metadata.timestamp[1] == next_timestamp:
            return sample
    return None  # no match found

BufferItem = namedtuple("BufferItem", ["batch", "is_rollout"])

class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, batch):
        self.buffer.append(batch)
    
    def sample(self, batch_size=1, mode="random"):
        """
        Sample a set of items from the replay buffer.
        
        Args:
            batch_size (int): The number of samples to return.
            mode (str): "random" to use random sampling (default),
                        "linear" to sample in the order of insertion.
                        
        Returns:
            List of sampled items.
        """
        if mode == "random":
            return random.sample(self.buffer, batch_size)
        elif mode == "linear":
            # Return the first batch_size items in the buffer in order.
            return list(self.buffer)[:batch_size]
        else:
            raise ValueError(f"Invalid sampling mode: {mode}")
    
    def __len__(self):
        return len(self.buffer)
    

@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    """
    Rollout-finetuning script using a PyTorch Lightning module with a Dataset.
    For scaling up we need TODO implement a (prioritize) buffer concept.

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

    # Setup config
    print(OmegaConf.to_yaml(cfg))
    # Set the internal precision for TensorCores (H100s)
    torch.set_float32_matmul_precision(cfg.training.precision_in)

    # Seed the experiment for numpy, torch and python.random.
    seed_everything(42, workers=True)

    # Load the Test Dataset
    print("Setting up Dataloader ...")
    data_dir = "data_small/2009_batches" # Needs a single batch item to start the rollouts from
    test_dataset = LargeClimateDataset(
        data_dir=data_dir, scaling_settings=cfg.data.scaling, num_species=cfg.data.species_number
    )
    test_dataset_seq = SequentialWindowDataset(test_dataset, cfg.finetune.rollout_steps)
    print("Reading test data from :", data_dir)
    test_dataloader = DataLoader(
        test_dataset_seq,
        batch_size=1,
        num_workers=cfg.training.workers,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
    )

    output_dir = HydraConfig.get().runtime.output_dir

    # Setup logger
    current_time = datetime.now()
    # log the metrics in the hydra folder (easier to find)
    mlf_logger_in_hydra_folder = MLFlowLogger(
        experiment_name="BFM_logs", run_name=f"BFM_{current_time}", save_dir=f"{output_dir}/logs"
    )
    # also log in the .mlruns folder so that you can run mlflow server and see every run together
    # tracking_uri = f"http://0.0.0.0:{cfg.mlflow.port}"
    mlf_logger_in_current_folder = MLFlowLogger(experiment_name="BFM_logs", run_name=f"BFM_{current_time}")

    checkpoint_path = cfg.evaluation.checkpoint_path
    # Load Model from Checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")

    device = torch.device(cfg.evaluation.test_device)
    print("weigths device", device)

    trainer = L.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        # log_every_n_steps=cfg.training.log_steps,
        # logger=[mlf_logger_in_hydra_folder, mlf_logger_in_current_folder],
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    loaded_model = BFM_Forecastinglighting.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        surface_vars=(["t2m", "msl"]),
        single_vars=(["lsm"]),
        atmos_vars=(["z", "t"]),
        species_vars=(["ExtinctionValue"]),
        species_distr_vars=(["Distribution"]),
        land_vars=(["Land", "NDVI"]),
        agriculture_vars=(["AgricultureLand", "AgricultureIrrLand", "ArableLand", "Cropland"]),
        forest_vars=(["Forest"]),
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
        ground_truth_dataset=test_dataset,
        strict=False,
        use_lora=True, # We finetune using LoRA
    )

    trainer.fit(loaded_model, test_dataloader)
if __name__ == "__main__":
    main()
