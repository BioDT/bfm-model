import torch

def print_dict_structure(d, parent_key="", nan_count=0, total_count=0):
    """
    Recursively print the structure, names, and shapes of tensors in a nested dictionary.
    Also count NaN values to calculate their percentage later.

    Arguments:
    - d: The dictionary (potentially nested) containing tensors or other dictionaries.
    - parent_key: A string to prefix variable names to keep track of hierarchy.
    - nan_count: Accumulated NaN count.
    - total_count: Accumulated total element count.
    
    Returns:
    nan_count, total_count
    """
    for key, value in d.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            # Print a header for this dictionary section
            print(f"\n[Exploring dictionary: {full_key}]")
            nan_count, total_count = print_dict_structure(value, full_key, nan_count, total_count)
        elif isinstance(value, torch.Tensor):
            # Print tensor name and shape
            shape_str = " x ".join(map(str, value.shape)) if value.dim() > 0 else "Scalar"
            print(f"Variable: {full_key} | Shape: {shape_str}")
            
            # Count NaNs in this tensor
            current_nan = torch.isnan(value).sum().item()
            current_total = value.numel()
            nan_count += current_nan
            total_count += current_total
        else:
            # Not a tensor or dict, print type info
            print(f"Variable: {full_key} | Type: {type(value)} (Not a tensor, skipping NaN count)")
    return nan_count, total_count


# Main execution:
if __name__ == "__main__":
    # 1) Load the dataset
    dataset = torch.load("data/batch_2000-01-14_00-00-00_to_2000-01-14_06-00-00.pt")  # Replace with the actual file path
    
    # 2) Go through the dictionary variables and subvariables and print their name and shapes
    print("\n=== Dataset Structure and Shapes ===")
    total_nan, total_elems = print_dict_structure(dataset)
    
    # 3) Count the nan variables and print the percentage vs the total number of variables
    # NaN percentage calculation:
    nan_percentage = (total_nan / total_elems * 100) if total_elems > 0 else 0
    
    # 4) Print results in a human-understandable format
    print("\n=== Summary of NaN Values ===")
    print(f"Total elements checked: {total_elems}")
    print(f"Total NaN values: {total_nan}")
    print(f"Percentage of NaN values: {nan_percentage:.2f}%")
