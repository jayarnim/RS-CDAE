from UTILS.constants import(
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
    COL_DICT,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
    SEED,
)

from typing import Literal

LOSS_FN_TYPE_POINTWISE = Literal["bce"]
METRIC_FN_TYPE = Literal["hr", "precision", "recall", "map", "ndcg"]