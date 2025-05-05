from pathlib import Path
from typing import Union
import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi

def convert_pt_to_safetensors(
    pt_file: Union[str, Path],
    safe_file: Union[str, Path] = None,
    state_key: str = "state_dict",
    strip_prefix: str = "",
) -> Path:
    """
    Convert any PyTorch/Lightning checkpoint (.pt|.ckpt) to safetensors.

    Args:
        pt_file: path to the pickle-based checkpoint.
        safe_file: desired output file; defaults to same stem + '.safetensors'.
        state_key: key that stores the weights inside a Lightning checkpoint.
                   Use None if the file is already just a state-dict.
        strip_prefix: optional prefix to drop (e.g. 'model.') so keys
                      match when you load into the original nn.Module.
    Returns:
        Path to the written .safetensors file.
    """
    pt_file = Path(pt_file)
    safe_file = Path(safe_file or pt_file.with_suffix(".safetensors"))

    checkpoint = torch.load(pt_file, map_location="cpu")

    if state_key and state_key in checkpoint:
        state_dict = checkpoint[state_key]
    else:
        state_dict = checkpoint

    if strip_prefix:
        state_dict = {
            k[len(strip_prefix):] if k.startswith(strip_prefix) else k: v
            for k, v in state_dict.items()
        }
    safe_path = str(safe_file)
    save_file(state_dict, safe_path)
    return safe_file, safe_path

def load_safetensors_into_model(
    safe_file: Union[str, Path],
    model: torch.nn.Module,
    device: str = "cpu",
    strict: bool = True,
):
    """
    Loads a .safetensors checkpoint into an nn.Module.

    Returns:
        (missing, unexpected) keys exactly like nn.Module.load_state_dict.
    """
    state_dict = load_file(str(safe_file), device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return missing, unexpected

if __name__ == "__main__":
    WEIGHTS_PATH = "/home/thanasis.trantas/git_projects/bfm-model/model_weights/epoch=02-train_loss=0.04.ckpt"
    file, path = convert_pt_to_safetensors(WEIGHTS_PATH)
    print(f"Done converting at {path}!")
    
    # HfApi().upload_file(
    #         repo_id="bfm/model/tba",
    #         path_or_fileobj=path,
    #         path_in_repo=safe_path.name,
    # )