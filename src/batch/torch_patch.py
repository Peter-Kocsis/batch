from typing import Optional, Dict, Union, Type, Tuple, Callable
from .batch import Batch

try:
    from torch.utils.data._utils.collate import default_collate_fn_map, collate


    def collate_batch_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        elem = batch[0]
        return Batch(**{key: collate([getattr(d, key) for d in batch], collate_fn_map=collate_fn_map)
                        for key in vars(elem) if not key.startswith("_")})


    default_collate_fn_map[Batch] = default_collate_fn_map
except ImportError:
    pass
