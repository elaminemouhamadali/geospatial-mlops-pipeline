# src/geo_mlops/core/execution/sharding.py

from __future__ import annotations

import math
from typing import List, Optional, Sequence, TypeVar

T = TypeVar("T")


def shard_sequence(
    items: Sequence[T],
    *,
    num_shards: Optional[int] = None,
    items_per_shard: Optional[int] = None,
    default_num_shards: Optional[int] = None,
    drop_empty: bool = True,
) -> List[List[T]]:
    """
    Split a sequence into deterministic shards.

    This utility is intentionally generic and should not know about evaluation,
    training, Ray, tiles, scenes, or task plugins.

    Priority:
      1. If items_per_shard is provided, use fixed-size chunks.
      2. Else use num_shards.
      3. Else use default_num_shards.
      4. Else use one shard.

    Examples:
        shard_sequence([1, 2, 3, 4, 5], items_per_shard=2)
        -> [[1, 2], [3, 4], [5]]

        shard_sequence([1, 2, 3, 4, 5], num_shards=2)
        -> [[1, 2, 3], [4, 5]]

    Args:
        items:
            Sequence of items to shard.
        num_shards:
            Desired number of shards.
        items_per_shard:
            Fixed number of items per shard. Overrides num_shards.
        default_num_shards:
            Fallback shard count when num_shards is None.
            For Ray, this can be total cluster CPUs.
        drop_empty:
            Whether to remove empty shards.

    Returns:
        List of shards, where each shard is a list of items.
    """

    item_list = list(items)

    if not item_list:
        return []

    if items_per_shard is not None:
        if items_per_shard <= 0:
            raise ValueError("items_per_shard must be > 0.")

        return [
            item_list[i : i + items_per_shard]
            for i in range(0, len(item_list), items_per_shard)
        ]

    if num_shards is None:
        num_shards = default_num_shards if default_num_shards is not None else 1

    if num_shards <= 0:
        raise ValueError("num_shards must be > 0.")

    num_shards = min(int(num_shards), len(item_list))

    shard_size = int(math.ceil(len(item_list) / num_shards))

    shards = [
        item_list[i : i + shard_size]
        for i in range(0, len(item_list), shard_size)
    ]

    if drop_empty:
        shards = [shard for shard in shards if shard]

    return shards


def shard_count(
    items: Sequence[T],
    *,
    num_shards: Optional[int] = None,
    items_per_shard: Optional[int] = None,
    default_num_shards: Optional[int] = None,
) -> int:
    """
    Convenience helper returning the number of shards that would be produced.
    """

    return len(
        shard_sequence(
            items,
            num_shards=num_shards,
            items_per_shard=items_per_shard,
            default_num_shards=default_num_shards,
        )
    )