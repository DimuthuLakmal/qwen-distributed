"""
Smoke test for the data pipeline: adapter -> SFTDataset -> SFTCollator ->
DataLoader, wired to the real configs (data.yaml/model.yaml/training.yaml)
via ConfigurationManager, using the actual trl-lib/llava-instruct-mix
dataset.

NOTE: On first run this downloads and caches the configured dataset split
via `load_dataset`, and the dataloader section uses num_workers=8 as
configured -- run this on a compute node (srun/sbatch), not the shared
login node.

Run from the repo root with:
    python -m pytest test/test_data_pipeline.py -v -s
or directly:
    python test/test_data_pipeline.py
"""

from pathlib import Path

from transformers import AutoTokenizer

from src.config.configuration import ConfigurationManager
from src.config.schema import DataLoaderConfig
from src.data.collate import SFTCollator
from src.data.dataloader import build_dataloader
from src.data.dataset import SFTDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _build_sft_dataset() -> tuple[SFTDataset, DataLoaderConfig]:
    config_manager = ConfigurationManager(
        data_config_path=CONFIGS_DIR / "data.yaml",
        model_config_path=CONFIGS_DIR / "model.yaml",
        training_config_path=CONFIGS_DIR / "training.yaml",
    )

    data_config = config_manager.get_data_config()
    dataloader_config = config_manager.get_dataloader_config()

    processor = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=data_config.cache_dir,
        local_files_only=True,
    )

    sft_dataset = SFTDataset.from_config(data_config, processor)

    return sft_dataset, dataloader_config


def test_dataset_produces_expected_fields():
    sft_dataset, _ = _build_sft_dataset()
    assert len(sft_dataset) > 0

    sample = sft_dataset[0]

    assert "input_ids" in sample
    assert "loss_mask" in sample
    assert sample["input_ids"].shape == sample["loss_mask"].shape
    assert sample["loss_mask"].any(), "expected at least one supervised token"

    print("\nsample input_ids shape:", sample["input_ids"].shape)
    print("sample supervised tokens:", int(sample["loss_mask"].sum()))
    if "pixel_values" in sample:
        print("sample pixel_values shape:", sample["pixel_values"].shape)
        print("sample image_grid_thw:", sample["image_grid_thw"])


def test_collator_pads_and_batches():
    sft_dataset, _ = _build_sft_dataset()

    collator = SFTCollator(pad_token_id=sft_dataset.tokenizer.pad_token_id)
    batch = collator([sft_dataset[0], sft_dataset[1]])

    batch_size, seq_len = batch["input_ids"].shape
    assert batch_size == 2
    assert batch["attention_mask"].shape == (batch_size, seq_len)
    assert batch["labels"].shape == (batch_size, seq_len)
    assert (batch["labels"] == -100).any()

    print("\nbatched input_ids shape:", batch["input_ids"].shape)
    if "pixel_values" in batch:
        print("batched pixel_values shape:", batch["pixel_values"].shape)
        print("batched image_grid_thw:", batch["image_grid_thw"])


def test_dataloader_yields_batches():
    sft_dataset, dataloader_config = _build_sft_dataset()

    dataloader, sampler = build_dataloader(sft_dataset, dataloader_config, shuffle=False)
    assert sampler is None  # single-process, no torch.distributed group initialized

    batch = next(iter(dataloader))
    assert batch["input_ids"].shape[0] == dataloader_config.batch_size

    print("\ndataloader batch input_ids shape:", batch["input_ids"].shape)


if __name__ == "__main__":
    test_dataset_produces_expected_fields()
    test_collator_pads_and_batches()
    test_dataloader_yields_batches()
    print("\nAll data pipeline smoke tests passed.")
