from typing import Optional
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from .utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    SEED,
)
from .data_splitter.python_splitters import python_stratified_split
from .dataloader import pointwise


class TRN_VAL_TST:
    def __init__(
        self, 
        n_users: int, 
        n_items: int,
        col_user: str=DEFAULT_USER_COL, 
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.col_user = col_user
        self.col_item = col_item

        self._set_up_components()

    def get(
        self, 
        origin: pd.DataFrame,
        trn_val_tst_ratio: dict=dict(trn=0.7, val=0.1, tst=0.2),
        neg_per_pos_ratio: dict=dict(trn=5, val=5, tst=99),
        batch_size: dict=dict(trn=256, val=256, tst=256),
        shuffle: bool=True,
        seed: int=SEED,
    ):
        kwargs = dict(
            trn_val_tst_ratio=trn_val_tst_ratio,
            neg_per_pos_ratio=neg_per_pos_ratio,
            batch_size=batch_size,
        )
        self._assert_arg_error(**kwargs)

        # split original data
        kwargs = dict(
            origin=origin,
            trn_val_tst_ratio=trn_val_tst_ratio,
            seed=seed,
        )
        split_dict = self._data_splitter(**kwargs)

        # generate data loaders
        loaders = []

        for split_type in ["trn", "val", "tst"]:
            kwargs = dict(
                origin=origin,
                split=split_dict[split_type], 
                neg_per_pos_ratio=neg_per_pos_ratio[split_type], 
                batch_size=batch_size[split_type], 
                shuffle=shuffle,
            )
            loader = self.dataloader.get(**kwargs)
            loaders.append(loader)

        return loaders

    def _data_splitter(
        self,
        origin: pd.DataFrame,
        trn_val_tst_ratio: dict,
        seed: int,
    ):
        split_type = list(trn_val_tst_ratio.keys())
        split_ratio = list(trn_val_tst_ratio.values())

        # trn_val_tst -> [trn, val, tst]
        kwargs = dict(
            data=origin,
            ratio=split_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=seed,
        )
        split_list = python_stratified_split(**kwargs)

        split_dict = dict(zip(split_type, split_list))

        return split_dict

    def _assert_arg_error(self, trn_val_tst_ratio, neg_per_pos_ratio, batch_size):
        CONDITION = (list(trn_val_tst_ratio.keys()) == ["trn", "val", "tst"])
        ERROR_MESSAGE = f"key of trn_val_tst_ratio must be ['trn', 'val', 'tst'], but: {list(trn_val_tst_ratio.keys())}"
        assert CONDITION, ERROR_MESSAGE

        CONDITION = (list(neg_per_pos_ratio.keys()) == ["trn", "val", "tst"])
        ERROR_MESSAGE = f"key of neg_per_pos_ratio must be ['trn', 'val', 'tst'], but: {list(neg_per_pos_ratio.keys())}"
        assert CONDITION, ERROR_MESSAGE

        CONDITION = (list(batch_size.keys()) == ["trn", "val", "tst"])
        ERROR_MESSAGE = f"key of batch_size must be ['trn', 'val', 'tst'], but: {list(batch_size.keys())}"
        assert CONDITION, ERROR_MESSAGE

    def _set_up_components(self):
        self._init_dataloader()

    def _init_dataloader(self):
        kwargs = dict(
            col_user=self.col_user,
            col_item=self.col_item,
        )
        self.dataloader = pointwise.CustomizedDataLoader(**kwargs)