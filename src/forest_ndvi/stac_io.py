from __future__ import annotations

from typing import List, Sequence, Tuple

import planetary_computer as pc
from pystac import Item
from pystac_client import Client

from .config import NDVIConfig


def get_stac_client() -> Client:
    """
    Return a Planetary Computer STAC client.
    """
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def _warn_if_empty(kind: str, date_range: Tuple[str, str]) -> None:
    """
    Raise a clear error when no items are returned by a STAC search.
    """
    raise RuntimeError(
        f"No {kind} items found for date range {date_range}. "
        "Try enlarging the time window, relaxing the cloud_cover_max "
        "constraint, or choosing a slightly different AOI."
    )


def search_sentinel2_items(
    cfg: NDVIConfig,
    bbox: Sequence[float],
    date_range: Tuple[str, str],
) -> List[Item]:
    """
    Search Sentinel-2 L2A items on Planetary Computer within a bbox and date range.

    Returns a signed list of STAC Items suitable for odc.stac.load.
    """
    client = get_stac_client()
    search = client.search(
        collections=[cfg.collection],
        bbox=bbox,
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": cfg.cloud_cover_max}},
        max_items=200,
    )

    items = list(search.items())

    if not items:
        _warn_if_empty("Sentinel-2", date_range)

    return [pc.sign(item) for item in items]


def search_worldcover_items(
    cfg: NDVIConfig,
    bbox: Sequence[float],
) -> List[Item]:
    """
    Search ESA WorldCover items for the given year.
    """
    client = get_stac_client()
    search = client.search(
        collections=[cfg.worldcover_collection],
        bbox=bbox,
        datetime=f"{cfg.worldcover_year}-01-01/{cfg.worldcover_year}-12-31",
        max_items=10,
    )

    items = list(search.items())

    if not items:
        _warn_if_empty(
            "WorldCover",
            (f"{cfg.worldcover_year}-01-01", f"{cfg.worldcover_year}-12-31"),
        )

    return [pc.sign(item) for item in items]
