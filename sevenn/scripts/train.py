from typing import Optional
import random

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model
from sevenn.sevenn_logger import Logger
from sevenn.train.trainer import Trainer

from .processing_epoch import processing_epoch


def loader_from_config(config, dataset, is_train=False):
    batch_size = config[KEY.BATCH_SIZE]
    shuffle = is_train and config[KEY.TRAIN_SHUFFLE]
    sampler = None
    if config[KEY.IS_DDP]:
        dist.barrier()
        sampler = DistributedSampler(
            dataset, dist.get_world_size(), 
            dist.get_rank(), shuffle=shuffle
        )
    return DataLoader(dataset, batch_size, shuffle, sampler=sampler)


def train_v2(config, working_dir: str):
    """
    Main program flow, since v0.9.6
    """
    from sevenn.train.graph_dataset import from_config
    from .processing_continue import processing_continue_v2
    Logger().timer_start('total')

    # config updated
    start_epoch = 1
    state_dicts: Optional[list[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        state_dicts, start_epoch = processing_continue_v2(config)

    datasets = from_config(config, working_dir)
    loaders = {
        k: loader_from_config(config, v, is_train=(k=='dataset'))
        for k, v in datasets.items()
    }

    Logger().write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    assert isinstance(model, Module)
    Logger().print_model_info(model, config)

    trainer = Trainer(model, config)
    if state_dicts:
        trainer.load_state_dicts(*state_dicts, strict=False)

    processing_epoch(
        trainer, config, loaders, start_epoch, init_csv, working_dir
    )
    Logger().timer_end('total', message='Total wall time')


def train(config, working_dir: str):
    """
    Main program flow, until v0.9.5
    """
    from .processing_dataset import processing_dataset
    from .processing_continue import processing_continue
    Logger().timer_start('total')

    # config updated
    state_dicts: Optional[list[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        state_dicts, start_epoch, init_csv = processing_continue(config)
    else:
        start_epoch, init_csv = 1, True

    # config updated
    train, valid, _ = processing_dataset(config, working_dir)
    datasets = {'dataset': train, 'validset': valid}
    loaders = {
        k: loader_from_config(config, v, is_train=(k == 'dataset'))
        for k, v in datasets.items()
    }
    loaders = list(loaders.values())

    Logger().write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    assert isinstance(model, Module)

    Logger().write('Model building was successful\n')

    trainer = Trainer(model, config)
    if state_dicts:
        trainer.load_state_dicts(*state_dicts, strict=False)

    Logger().print_model_info(model, config)

    Logger().write('Trainer initialized, ready to training\n')
    Logger().bar()

    processing_epoch(
        trainer, config, loaders, start_epoch, init_csv, working_dir
    )
    Logger().timer_end('total', message='Total wall time')
