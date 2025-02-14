import os
import glob


def load_last_run_id(run_id_file: str):
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            return f.read().strip()
    return None

def save_run_id(run_id_file: str, run_id: str):
    with open(run_id_file, "w") as f:
        f.write(run_id)

def print_auto_logged_info(r, mlflow_client):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in mlflow_client.list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def get_latest_checkpoint(ckpt_dir, pattern="*.ckpt"):
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, pattern))
    if not checkpoint_files:
        return None
    # Choose the checkpoint with the latest creation time
    latest_ckpt = max(checkpoint_files, key=os.path.getctime)
    return latest_ckpt