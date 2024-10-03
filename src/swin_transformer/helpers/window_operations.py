import itertools
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange


def window_partition_and_reverse_3d(
    x: torch.Tensor,
    window_size: tuple[int, int, int],
    mode: Literal["partition", "reverse"] = "partition",  # Default: 'partition', can be 'partition' or 'reverse'
    total_channels: int = None,  # Required for reconstruction mode
    total_height: int = None,  # Required for reconstruction mode
    total_width: int = None,  # Required for reconstruction mode
) -> torch.Tensor:
    """Partitions a 3D tensor into windows or reconstructs the original tensor from partitioned windows.

    This function can either partition a 5D tensor into smaller 3D windows or reconstruct the original tensor
    from its partitioned windows, depending on the provided mode. It is commonly used in 3D vision transformers
    to process local regions independently and to combine local regions back into a global representation.

    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W, D] for partitioning or
                          [num_windows * B, Wc, Wh, Ww, D] for reconstruction.
        window_size (tuple[int, int, int]): 3D window size [Wc, Wh, Ww].
        mode (str): Operation mode, 'partition' to partition the tensor, 'reverse' to reconstruct it.
                    Default: 'partition'.
        total_channels (int, optional): Total number of channels in the original input. Required for reconstruction.
        total_height (int, optional): Height of the original input. Required for reconstruction.
        total_width (int, optional): Width of the original input. Required for reconstruction.

    Returns:
        torch.Tensor: If partitioning, returns tensor of shape [num_windows * B, Wc, Wh, Ww, D].
                      If reconstructing, returns tensor of shape [B, C, H, W, D].

    Raises:
        AssertionError: If input dimensions are not divisible by the corresponding window size or if reconstruction
                        is attempted without providing total dimensions.
    """
    if mode == "partition":
        # Partitioning case
        B, C, H, W, D = x.shape  # Extract dimensions from input tensor
        Wc, Wh, Ww = window_size  # Unpack window size

        # Ensure input dimensions are divisible by window size
        assert C % Wc == 0, f"C ({C}) must be divisible by window size ({Wc})."
        assert H % Wh == 0, f"H ({H}) must be divisible by window size ({Wh})."
        assert W % Ww == 0, f"W ({W}) must be divisible by window size ({Ww})."

        # Reshape the input tensor to separate window dimensions
        x_reshaped = x.view(B, C // Wc, Wc, H // Wh, Wh, W // Ww, Ww, D)  # shape: [B, C//Wc, Wc, H//Wh, Wh, W//Ww, Ww, D]

        # Rearrange dimensions to group windows together
        windows = rearrange(
            x_reshaped, "B C1 Wc H1 Wh W1 Ww D -> (B C1 H1 W1) Wc Wh Ww D"
        )  # shape: [total_num_windows * batch_size, window_channels, window_height, window_width, depth]

        return windows  # Return partitioned windows

    elif mode == "reverse":
        # Reconstruction case
        # Ensure that the total dimensions are divisible by the corresponding window sizes
        assert total_channels is not None, "Total channels must be provided for reconstruction."
        assert total_height is not None, "Total height must be provided for reconstruction."
        assert total_width is not None, "Total width must be provided for reconstruction."

        assert (
            total_channels % window_size[0] == 0
        ), f"Total channels ({total_channels}) must be divisible by window size ({window_size[0]})."
        assert (
            total_height % window_size[1] == 0
        ), f"Total height ({total_height}) must be divisible by window size ({window_size[1]})."
        assert (
            total_width % window_size[2] == 0
        ), f"Total width ({total_width}) must be divisible by window size ({window_size[2]})."

        # Calculate the number of windows in each dimension
        num_windows_channels = total_channels // window_size[0]  # of channel windows
        num_windows_height = total_height // window_size[1]  # of height windows
        num_windows_width = total_width // window_size[2]  # of width windows

        # Calculate the batch size from the shape of the partitioned windows
        batch_size = int(x.shape[0] / (num_windows_channels * num_windows_height * num_windows_width))

        # Rearrange the partitioned windows back into the original tensor shape
        reconstructed_tensor = rearrange(
            x,
            "(B C1 H1 W1) Wc Wh Ww D -> B (C1 Wc) (H1 Wh) (W1 Ww) D",
            B=batch_size,
            C1=num_windows_channels,
            H1=num_windows_height,
            W1=num_windows_width,
            Wc=window_size[0],
            Wh=window_size[1],
            Ww=window_size[2],
        )  # shape: [B, C, H, W, D]

        return reconstructed_tensor
    else:
        raise ValueError(f"Invalid mode. Choose either 'partition' or 'reverse', got {mode}.")


def apply_or_remove_3d_padding(
    x: torch.Tensor, pad_size: tuple[int, int, int], mode: Literal["apply", "remove"] = "apply", pad_value: float = 0.0
) -> torch.Tensor:
    """
    Applies or removes 3D padding to/from the input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape [B, C, H, W, D].
        pad_size (tuple[int, int, int]): Padding sizes for each dimension [C_pad, H_pad, W_pad].
        mode (str): 'apply' to add padding, 'remove' to crop padding. Default: 'apply'
        pad_value (float): Value used for padding when mode is 'apply'. Default: 0.0

    Returns:
        torch.Tensor: Padded or cropped tensor.
    """
    assert mode in ["apply", "remove"], "Mode must be either 'apply' or 'remove'"

    def _calculate_3d_padding(pad_size: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
        """
        Calculates padding for each side in 3D.

        Args:
            pad_size (tuple[int, int, int]): Padding sizes [C_pad, H_pad, W_pad].

        Returns:
            tuple[int, int, int, int, int, int]: Padding values [left, right, top, bottom, front, back].
        """
        C_pad, H_pad, W_pad = pad_size
        assert all(p >= 0 for p in pad_size), f"All padding values must be non-negative, got {pad_size}"

        # Calculate padding for each dimension
        pad_left, pad_right = W_pad // 2, W_pad - W_pad // 2
        pad_top, pad_bottom = H_pad // 2, H_pad - H_pad // 2
        pad_front, pad_back = C_pad // 2, C_pad - C_pad // 2

        return pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back

    padding = _calculate_3d_padding(pad_size)

    if mode == "apply":
        # Apply padding to the input tensor
        return F.pad(x, (0, 0, *padding[::-1]), value=pad_value)  # shape: [B, C+C_pad, H+H_pad, W+W_pad, D]
    else:  # mode == 'remove'
        # Remove padding from the input tensor
        B, C, H, W, D = x.shape
        pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = padding
        return x[
            :, pad_front : C - pad_back, pad_top : H - pad_bottom, pad_left : W - pad_right, :
        ]  # shape: [B, C-C_pad, H-H_pad, W-W_pad, D]


def compute_3d_merge_groups() -> list[tuple[int, int]]:
    """
    Computes the groups to be merged for 3D shifted window attention to achieve left-right connectivity.

    This function is crucial for maintaining global context in 3D data, especially for spherical or periodic data.

    Returns:
        list[tuple[int, int]]: List of group pairs to be merged in 3D space.
    """
    # Define 2D merge groups for a single slice
    merge_groups_2d = torch.tensor([(1, 2), (4, 5), (7, 8)], dtype=torch.int)

    # Create offsets for each depth slice
    slice_offsets = torch.arange(0, 27, 9, dtype=torch.int).view(-1, 1, 1)  # shape: [3, 1, 1]

    # Extend 2D merge groups to 3D by adding offsets
    merge_groups_3d = merge_groups_2d + slice_offsets

    # Reshape to a list of tuples
    merge_groups_3d = merge_groups_3d.view(-1, 2).tolist()

    return merge_groups_3d


@lru_cache
def compute_3d_shifted_window_mask(
    C: int,
    H: int,
    W: int,
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    warped: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the mask for 3D shifted-window attention.

    Args:
        C (int): Number of channels (depth).
        H (int): Height of the image.
        W (int): Width of the image.
        window_size (tuple[int, int, int]): Window sizes [Wc, Wh, Ww].
        shift_size (tuple[int, int, int]): Shift sizes [Sc, Sh, Sw].
        device (torch.device): Computation device.
        dtype (torch.dtype): Data type of the mask. Default: torch.bfloat16
        warped (bool): If True, assume left and right sides of the image are connected. Default: True

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Attention mask for each window. Masked entries are -100 and non-masked are 0.
            - Image mask splitting input patches into groups (for debugging).
    """
    # Initialize image mask
    img_mask = torch.zeros([1, C, H, W, 1], device=device, dtype=dtype)

    # Define slices for each dimension, these slices are used to split the image into windows, and will serve as indices for the image mask
    c_slices = (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None))
    h_slices = (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None))
    w_slices = (slice(0, -window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None))

    # Assign group numbers to different regions of the image
    for group_num, (c, h, w) in enumerate(
        itertools.product(c_slices, h_slices, w_slices)
    ):  # iterate over all slices, this will make Cartesian products of the slices
        img_mask[:, c, h, w, :] = group_num

    if warped:
        # Merge groups for warped (cyclic) attention
        for group1, group2 in compute_3d_merge_groups():
            img_mask = torch.where(img_mask == group1, group2, img_mask)

    # Calculate padding to make dimensions divisible by window size
    pad_size = tuple((window_size[i] - dim % window_size[i]) % window_size[i] for i, dim in enumerate([C, H, W]))

    # Aapply padding to the image mask
    img_mask_padded = apply_or_remove_3d_padding(
        img_mask, pad_size, mode="apply", pad_value=group_num + 1
    )  # shape: [1, C+pad_C, H+pad_H, W+pad_W, 1]

    # Partition the mask into windows
    mask_windows = window_partition_and_reverse_3d(
        img_mask_padded, window_size, mode="partition"
    )  # shape: [num_windows, Wc, Wh, Ww, 1]

    # Flatten each window
    mask_windows_flat = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])  # shape: [num_windows, Wc*Wh*Ww]

    # Compute pairwise differences between windows
    attn_mask = mask_windows_flat.unsqueeze(1) - mask_windows_flat.unsqueeze(2)  # shape: [num_windows, num_windows, Wc*Wh*Ww]

    # Convert differences to binary attention mask
    attn_mask = torch.where(
        attn_mask != 0, torch.tensor(-100.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
    )  # shape: [num_windows, num_windows, Wc*Wh*Ww]

    return attn_mask, img_mask
