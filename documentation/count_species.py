import hydra
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
import pickle

from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from bfm_model.bfm.dataloder import LargeClimateDataset, custom_collate
from bfm_model.bfm.utils import compute_species_occurrences, plot_species_stats_from_lists, plot_species_stats_from_single_lists


hydra.initialize(config_path="", version_base=None)
cfg = hydra.compose(config_name="viz_config")
print(OmegaConf.to_yaml(cfg))

cfg = hydra.compose(config_name="viz_config.yaml", overrides=["data.scaling.enabled=False",])

if __name__ == '__main__':
    dataset = LargeClimateDataset(data_dir=cfg.data.data_path, scaling_settings=cfg.data.scaling ,num_species=cfg.data.species_number)

    stats_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=15,
        collate_fn=custom_collate,
        drop_last=True,
        shuffle=False,
    )

    sample_stats_origin = []
    for data_orign in stats_dataloader:
        species_stats_origin = compute_species_occurrences(data_orign)
        sample_stats_origin.append(species_stats_origin)

    with open('species_stats_train.pkl', 'wb') as file:
        pickle.dump(sample_stats_origin, file)