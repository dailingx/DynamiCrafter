import os, sys, datetime, glob
import logging
import argparse
from functools import partial
from packaging import version
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from pytorch_lightning.plugins import DDPPlugin

from utils.utils import instantiate_from_config
from utils.common_utils import str2bool


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None,
                 shuffle_test_loader=False, shuffle_val_dataloader=False,
                 use_worker_init_fn=False,
                 test_max_n_samples=None, val_max_n_samples=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap
        self.test_max_n_samples = test_max_n_samples
        self.val_max_n_samples = val_max_n_samples

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(f'DataModuleFromConfig stage:{stage}')
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        loader = DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=None, collate_fn=None,
                          )
        return loader

    def _val_dataloader(self, shuffle=False):
        if self.val_max_n_samples is not None:
            dataset = torch.utils.data.Subset(self.datasets["validation"], list(range(self.val_max_n_samples)))
        else:
            dataset = self.datasets["validation"]
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=None,
                          shuffle=shuffle,
                          collate_fn=None,
                          )

    def _test_dataloader(self, shuffle=False):
        if self.test_max_n_samples is not None:
            dataset = torch.utils.data.Subset(self.datasets["test"], list(range(self.test_max_n_samples)))
        else:
            dataset = self.datasets["test"]
        return DataLoader(dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=None, shuffle=shuffle,
                          collate_fn=None,
                          )

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=None,
                          collate_fn=None,
                          )
