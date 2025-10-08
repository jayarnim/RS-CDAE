from typing import Optional, Literal
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class CustomizedDataset(Dataset):
    def __init__(
        self,
        origin: pd.DataFrame,
        split: pd.DataFrame,
        neg_per_pos_ratio: Optional[int]=None,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.origin = origin
        self.split = split
        self.neg_per_pos_ratio = neg_per_pos_ratio
        self.col_user = col_user
        self.col_item = col_item

        self._set_up_components()

    def __len__(self):
        return self.n_user

    def __getitem__(self, idx):
        user_idx = self.user_list[idx]
        pos_idx = self.pos_per_user_split.get(user_idx, set())

        x = torch.zeros(self.n_item, dtype=torch.float32)
        x[list(pos_idx)] = 1.0

        mask = self._mask(user_idx, pos_idx)

        return user_idx, x, mask

    def _mask(self, user_idx, pos_idx):
        # generate mask
        mask = torch.zeros(self.n_item, dtype=torch.float32)

        # pos mask
        mask[list(pos_idx)] = 1.0

        # neg mask
        neg_candidates = self.neg_per_user.get(user_idx, [])
        
        if self.neg_per_pos_ratio is None:
            n_neg = len(neg_candidates)
        else:
            n_neg = min(len(neg_candidates), len(pos_idx) * self.neg_per_pos_ratio)
        
        sampled_neg = random.sample(neg_candidates, n_neg)
        mask[sampled_neg] = 1.0

        return mask

    def _set_up_components(self):
        self.user_list = sorted(self.origin[self.col_user].unique())
        self.item_list = sorted(self.origin[self.col_item].unique())

        self.n_user = len(self.user_list)
        self.n_item = len(self.item_list)

        pos_per_user = {
            user: set(self.origin.loc[self.origin[self.col_user] == user, self.col_item].tolist())
            for user in self.user_list
        }

        self.neg_per_user = {
            user: list(set(self.item_list) - pos_per_user[user])
            for user in self.user_list
        }

        self.pos_per_user_split = {
            user: set(self.split.loc[self.split[self.col_user]==user, self.col_item].tolist())
            for user in self.user_list
        }

class CustomizedDataLoader:
    def __init__(
        self,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.col_user = col_user
        self.col_item = col_item

    def get(
        self,
        origin: pd.DataFrame,
        split: pd.DataFrame,
        neg_per_pos_ratio: Optional[int]=None,
        batch_size: int=256,
        shuffle: bool=True,
    ):
        dataset = CustomizedDataset(
            origin=origin,
            split=split,
            neg_per_pos_ratio=neg_per_pos_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
        )
        return loader

    def _collate(self, batch):
        user_list, x_list, mask_list = zip(*batch)
        user_idx_batch = torch.tensor(user_list, dtype=torch.long)
        x_batch    = torch.stack(x_list, dim=0)
        mask_batch = torch.stack(mask_list, dim=0)
        return user_idx_batch, x_batch, mask_batch