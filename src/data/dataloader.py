from typing import Optional, Type, Union

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.config.schema import DataLoaderConfig
from src.data.collate import SFTCollator
from src.data.eval_collate import GenerationCollator


def build_dataloader(
    dataset: Dataset,
    config: DataLoaderConfig,
    shuffle: bool = True,
    collator_cls: Union[Type[SFTCollator], Type[GenerationCollator]] = SFTCollator,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Build the DataLoader for an SFTDataset (or SFTEvalDataset).

    When torch.distributed is initialized (DDP/FSDP), each rank gets a
    non-overlapping shard of the dataset via DistributedSampler instead
    of every rank iterating over the full dataset.

    The caller must call sampler.set_epoch(epoch) before each epoch so
    that shuffling differs across epochs; the returned sampler is None
    when running single-process.

    collator_cls is pluggable (e.g. GenerationCollator for eval) but
    must accept the same `pad_token_id` constructor argument as
    SFTCollator.
    """

    collator = collator_cls(pad_token_id=dataset.tokenizer.pad_token_id)

    sampler: Optional[DistributedSampler] = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=config.drop_last,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        drop_last=config.drop_last,
    )

    return dataloader, sampler
